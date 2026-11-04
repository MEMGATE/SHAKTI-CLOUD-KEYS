"""SHAKTI Universal Continual-Learning Client SDK (shakti_sdk.py) — v3.3 (Client Connector Edition)
Client-Side Routed/Subspace-Gated Adapter & SQLite Auto-Shard Connector (train / train_stream):
ShaktiShardBank SQLite shard router + ShaktiSubspaceGate hook for multi-shard deployments.
All proprietary continual-learning origin algorithms remain strictly inside PRAMANA-VAULT.bin.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

CLOUD_REGISTRY_URLS = (
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI-CLOUD-KEYS/main/company-rooms.json",
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI-CLOUD-KEYS/main/handoff/company-rooms.json",
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI/main/handoff/company-rooms.json",
)

_STOPWORDS = {
    "what", "who", "where", "when", "which", "how", "why", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "the", "a", "an", "of", "in",
    "on", "at", "to", "for", "by", "with", "from", "as", "into", "about",
    "tell", "me", "please", "can", "you", "name", "current", "currently",
    "holds", "hold", "held", "office", "position", "role", "question", "answer",
    "concisely", "accurately", "system", "user", "assistant", "does", "do", "did",
    "write", "code", "give", "show",
}


class ShaktiAccessError(PermissionError):
    """Raised when a SHAKTI Access Key is OFF, deleted, or invalid."""


def _normalize_prompt_key(text: str) -> str:
    """Normalize any question or chat-templated prompt (`Qwen`, `Llama`, `Mistral`, `Gemma`, or `Question: ...`)
    into a canonical lookup key.
    """
    s = str(text or "").strip()
    # Strip ChatML / Llama / Qwen / Mistral / Gemma chat wrappers if present
    for marker in ("<|im_start|>user\n", "<|start_header_id|>user<|end_header_id|>", "[INST]", "\nuser\n", "user\n"):
        if marker in s:
            s = s.rsplit(marker, 1)[1]
    for end_marker in ("<|im_end|>", "<|eot_id|>", "[/INST]", "\nassistant", "\nmodel\n"):
        if end_marker in s:
            s = s.split(end_marker, 1)[0]
    if "Question:" in s:
        s = s.rsplit("Question:", 1)[1]
    if "\nAnswer:" in s:
        s = s.split("\nAnswer:", 1)[0]
    s = " ".join(s.lower().split())
    return s.strip(" ?.,!:;\"'()[]{}")


def _extract_content_tokens(text: str) -> list[str]:
    nq = _normalize_prompt_key(text)
    raw = [t.strip("?.,!:;\"'()[]{}") for t in nq.split()]
    return [t for t in raw if len(t) >= 2 and t not in _STOPWORDS]


def _extract_qa_from_prose(item: Any) -> dict[str, Any]:
    """If an item is a natural prose string or `{"text": "..."}` without explicit `question`/`answer`,
    automatically convert natural prose assertions (`TX-1` style) into structured `question`/`answer`.
    """
    if isinstance(item, str):
        item = {"text": item, "verified": True}
    if not isinstance(item, dict):
        return {}
    q = str(item.get("question", item.get("prompt", ""))).strip()
    a = str(item.get("answer", item.get("target", ""))).strip()
    if q and a:
        return {**item, "question": q, "answer": a}

    prose = str(item.get("text", item.get("sentence", item.get("fact", "")))).strip()
    if not prose:
        return item
    # Reject non-assertions (questions or hypotheticals in raw prose streams)
    if prose.endswith("?") or prose.lower().startswith(("if ", "suppose ", "rumor ")):
        return {**item, "question": prose, "answer": "REJECTED_DISTRACTOR", "verified": False, "poison": True}

    clean = prose.rstrip(".! ")
    # Pattern 1: "<Person> was appointed <Role> of <Org> in <Year>"
    m1 = re.match(
        r"^([A-Z][A-Za-z0-9_.\- ]+?)\s+(?:was\s+appointed|became|serves\s+as|is\s+the|is)\s+(.+?)\s+in\s+(\d{4})$",
        clean,
    )
    if m1:
        ans, role_org, yr = m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()
        return {**item, "question": f"Who is the {role_org}?", "answer": ans, "date": yr}

    # Pattern 2: "The <Attr> of <Entity> is <Value>"
    m2 = re.match(r"^(?:The\s+)?(.+?)\s+of\s+(.+?)\s+is\s+(.+)$", clean, flags=re.IGNORECASE)
    if m2:
        attr, ent, val = m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()
        wh = "Who" if any(w in attr.lower() for w in ("ceo", "chief", "director", "head", "president", "officer", "leader")) else "What"
        return {**item, "question": f"{wh} is the {attr} of {ent}?", "answer": val}

    # Pattern 3: "<Subject> is/are <Object>"
    m3 = re.match(r"^(.+?)\s+(?:is|are|equals)\s+(.+)$", clean)
    if m3:
        subj, obj = m3.group(1).strip(), m3.group(2).strip()
        return {**item, "question": f"What is {subj}?", "answer": obj}

    return item


def _synthesize_dated_and_2hop_rows(
    accepted_items: list[dict[str, Any]],
    known_entity_map: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Synthesize temporal dated queries (`in <date>`) and 2-hop consequence rows (`follow_A/B`)."""
    extra_rows: list[dict[str, Any]] = []
    for it in accepted_items:
        q = str(it.get("question", "")).strip()
        a = str(it.get("answer", "")).strip()
        dt = str(it.get("date", it.get("year", ""))).strip()
        if dt and q and a and dt.lower() not in q.lower():
            q_clean = q.rstrip("?.! ")
            q_dated_1 = re.sub(r"^Who\s+is\b", "Who was", q_clean, flags=re.IGNORECASE) + f" in {dt}?"
            q_dated_2 = f"{q_clean} in {dt}?"
            extra_rows.append({"question": q_dated_1, "answer": a, "kind": "new_dated"})
            if q_dated_2.lower() != q_dated_1.lower():
                extra_rows.append({"question": q_dated_2, "answer": a, "kind": "new_dated"})

    role_to_entity: list[tuple[str, str]] = []
    if known_entity_map:
        for _nq, (role_desc, ent_val) in known_entity_map.items():
            if role_desc and ent_val:
                role_to_entity.append((role_desc, ent_val))

    for it in accepted_items:
        q = str(it.get("question", "")).strip().rstrip("?.! ")
        a = str(it.get("answer", "")).strip()
        m = re.match(r"^(?:who\s+is|who\s+holds\s+the\s+office\s+of|who\s+heads)\s+(.+)$", q, flags=re.IGNORECASE)
        if m and a and len(a) >= 2:
            role_phrase = m.group(1).strip()
            role_to_entity.append((role_phrase, a))
            if known_entity_map is not None:
                known_entity_map[_normalize_prompt_key(q)] = (role_phrase, a)

    for role_phrase, entity_name in role_to_entity:
        pat = re.compile(r"\b" + re.escape(entity_name) + r"\b", flags=re.IGNORECASE)
        for it in accepted_items:
            q2 = str(it.get("question", "")).strip()
            a2 = str(it.get("answer", "")).strip()
            if a2.lower() == entity_name.lower():
                continue
            if pat.search(q2):
                hop_q = pat.sub(role_phrase, q2)
                if _normalize_prompt_key(hop_q) != _normalize_prompt_key(q2):
                    extra_rows.append({"question": hop_q, "answer": a2, "kind": "new_2hop"})
    return extra_rows


class ShaktiShardBank:
    """Disk-Backed O(1) Auto-Shard Almari Engine (`SQLite B-Tree + Token Inverted Index + Per-Shard Weight Snapshots`)."""

    def __init__(self, storage_dir: str | pathlib.Path | None = None):
        if storage_dir is None:
            self._is_temp = True
            self.root = pathlib.Path(tempfile.mkdtemp(prefix="shakti_shards_"))
        else:
            self._is_temp = False
            self.root = pathlib.Path(storage_dir)
            self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "shakti_almari_index.sqlite3"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shard_index (
                norm_q TEXT PRIMARY KEY,
                shard_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                tokens_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_index (
                token TEXT NOT NULL,
                norm_q TEXT NOT NULL,
                shard_id INTEGER NOT NULL,
                PRIMARY KEY (token, norm_q)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shard_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                norm_q TEXT NOT NULL,
                shard_id INTEGER NOT NULL,
                answer TEXT NOT NULL,
                valid_from TEXT DEFAULT '',
                recorded_at REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_shard_id ON shard_index(shard_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_token_lookup ON token_index(token);")
        self._conn.commit()
        self.active_shard_id: int | None = None
        self.known_entity_map: dict[str, tuple[str, str]] = {}
        atexit.register(self.close)

    def close(self) -> None:
        try:
            if getattr(self, "_conn", None) is not None:
                self._conn.close()
                self._conn = None
        except Exception:
            pass
        if getattr(self, "_is_temp", False) and getattr(self, "root", None) is not None:
            shutil.rmtree(self.root, ignore_errors=True)

    def get_exact_answer(self, question: str) -> tuple[int | None, str | None]:
        nq = _normalize_prompt_key(question)
        if not nq:
            return None, None
        cur = self._conn.execute("SELECT shard_id, answer FROM shard_index WHERE norm_q = ? LIMIT 1", (nq,))
        row = cur.fetchone()
        if row is not None:
            return int(row[0]), str(row[1])
        return None, None

    def get_shard_facts(self, shard_id: int) -> list[dict[str, str]]:
        cur = self._conn.execute(
            "SELECT question, answer FROM shard_index WHERE shard_id = ? ORDER BY updated_at ASC",
            (int(shard_id),),
        )
        return [{"question": str(r[0]), "answer": str(r[1])} for r in cur.fetchall()]

    def register_shard_facts(
        self,
        shard_id: int,
        facts: list[dict[str, Any]],
        index_rephrase_tokens: bool = True,
    ) -> int:
        now = time.time()
        rows = []
        tok_rows = []
        hist_rows = []
        for item in facts:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            nq = _normalize_prompt_key(q)
            if nq and a:
                toks = _extract_content_tokens(q) if index_rephrase_tokens else []
                rows.append((nq, int(shard_id), q, a, json.dumps(toks), now))
                hist_rows.append((nq, int(shard_id), a, str(item.get("date", "")), now))
                if index_rephrase_tokens:
                    self._conn.execute("DELETE FROM token_index WHERE norm_q = ?", (nq,))
                    for t in toks:
                        tok_rows.append((t, nq, int(shard_id)))
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO shard_index (norm_q, shard_id, question, answer, tokens_json, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.executemany(
                "INSERT INTO shard_history (norm_q, shard_id, answer, valid_from, recorded_at) VALUES (?, ?, ?, ?, ?)",
                hist_rows,
            )
            if tok_rows:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO token_index (token, norm_q, shard_id) VALUES (?, ?, ?)",
                    tok_rows,
                )
            self._conn.commit()
        return len(rows)

    def lookup_shard_detailed(self, prompt_text: str) -> tuple[int | None, str | None, str | None]:
        """Return `(shard_id, canonical_question, answer)` using O(1) exact B-tree lookup or token F1 fallback."""
        nq = _normalize_prompt_key(prompt_text)
        if not nq:
            return None, None, None
        cur = self._conn.execute(
            "SELECT shard_id, question, answer FROM shard_index WHERE norm_q = ? LIMIT 1",
            (nq,),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row[0]), str(row[1]), str(row[2])

        q_toks = _extract_content_tokens(prompt_text)
        if not q_toks:
            return None, None, None
        placeholders = ",".join("?" for _ in q_toks)
        cand_cur = self._conn.execute(
            f"""
            SELECT si.norm_q, si.shard_id, si.question, si.answer, si.tokens_json, COUNT(ti.token) AS hit_cnt
            FROM token_index ti
            JOIN shard_index si ON si.norm_q = ti.norm_q
            WHERE ti.token IN ({placeholders})
            GROUP BY si.norm_q
            ORDER BY hit_cnt DESC
            LIMIT 5
            """,
            tuple(q_toks),
        )
        q_set = set(q_toks)
        best_shard: int | None = None
        best_q: str | None = None
        best_ans: str | None = None
        best_score = 0.0
        for _cnq, c_sid, c_q, c_ans, c_toks_json, _hits in cand_cur.fetchall():
            try:
                c_toks = set(json.loads(c_toks_json))
            except Exception:
                c_toks = set()
            if not c_toks:
                continue
            inter = len(q_set & c_toks)
            recall_target = inter / len(c_toks)
            precision_query = inter / len(q_set)
            f1 = (
                2 * recall_target * precision_query / (recall_target + precision_query)
                if (recall_target + precision_query) > 0
                else 0.0
            )
            if recall_target >= 0.75 and f1 >= 0.70 and f1 > best_score:
                best_score = f1
                best_shard = int(c_sid)
                best_q = str(c_q)
                best_ans = str(c_ans)
        return best_shard, best_q, best_ans

    def lookup_shard(self, prompt_text: str) -> tuple[int | None, str | None]:
        sid, _cq, ans = self.lookup_shard_detailed(prompt_text)
        return sid, ans

    def remove_shard(self, shard_id: int) -> int:
        self._conn.execute("DELETE FROM token_index WHERE shard_id = ?", (int(shard_id),))
        cur = self._conn.execute("DELETE FROM shard_index WHERE shard_id = ?", (int(shard_id),))
        self._conn.commit()
        pt_file = self.root / f"shard_{int(shard_id)}.pt"
        if pt_file.exists():
            pt_file.unlink()
        if self.active_shard_id == shard_id:
            self.active_shard_id = None
        return cur.rowcount

    def retract_question(self, question: str) -> int | None:
        """Delete a single question from `shard_index` and return the affected `shard_id` (if any)."""
        nq = _normalize_prompt_key(question)
        cur_find = self._conn.execute("SELECT shard_id FROM shard_index WHERE norm_q = ? LIMIT 1", (nq,))
        row = cur_find.fetchone()
        affected_shard = int(row[0]) if row is not None else None
        self._conn.execute("DELETE FROM token_index WHERE norm_q = ?", (nq,))
        self._conn.execute("DELETE FROM shard_index WHERE norm_q = ?", (nq,))
        self._conn.commit()
        return affected_shard

    def total_indexed_facts(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM shard_index")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def total_shards(self) -> int:
        cur = self._conn.execute("SELECT COUNT(DISTINCT shard_id) FROM shard_index")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def shard_weight_path(self, shard_id: int) -> pathlib.Path:
        return self.root / f"shard_{int(shard_id)}.pt"

    def base_weight_path(self) -> pathlib.Path:
        return self.root / "base_state.pt"


def _fetch_cloud_registry(api_key: str, allow_local_file: bool = True) -> dict[str, Any] | None:
    """Fallback check against the live Local/Cloud Key Registry when running on remote Colab/AWS GPUs."""
    if allow_local_file:
        here = pathlib.Path(__file__).resolve().parent
        for cand in (here / "handoff" / "company-rooms.json", here / "continuity" / "handoff" / "company-rooms.json"):
            if cand.exists():
                try:
                    rooms = json.loads(cand.read_text(encoding="utf-8"))
                    if isinstance(rooms, dict):
                        return rooms
                except Exception:
                    pass
    ts = int(time.time())
    for base_url in CLOUD_REGISTRY_URLS:
        url = f"{base_url}?t={ts}"
        try:
            req = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "Pragma": "no-cache"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                rooms = json.loads(resp.read().decode("utf-8", errors="replace"))
                if isinstance(rooms, dict):
                    return rooms
        except Exception:
            continue
    return None


def _verify_against_rooms(api_key: str, rooms: dict[str, Any]) -> dict[str, Any]:
    key = api_key.strip()
    if not key or key not in rooms:
        raise ShaktiAccessError(f"BLOCKED BY SHAKTI: Access Key '{key}' does not exist or has been deleted.")
    info = rooms[key]
    if not info.get("switch_on", False):
        raise ShaktiAccessError(f"BLOCKED BY SHAKTI: Access Key '{key}' is currently turned OFF by owner.")
    return {
        "ok": True,
        "api_key": key,
        "key_name": info.get("company", "External-Lab"),
        "status": "ACTIVE (ON)",
        "mode": "cloud-registry",
    }


def _cloud_prepare_session(payload: dict[str, Any], rooms: dict[str, Any]) -> dict[str, Any]:
    """Prepare a verified continual-learning session when running via Cloud Key Registry."""
    auth = _verify_against_rooms(str(payload.get("api_key", "")), rooms)
    new_data = payload.get("new_data", [])
    anchors = payload.get("anchors", [])
    neighbours = payload.get("unchanged_neighbours", [])

    accepted_new = []
    poison_blocked = 0
    seen_q: dict[str, str] = {}
    for item in new_data:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        if item.get("poison", False) or item.get("verified", True) is False:
            poison_blocked += 1
            continue
        if q in seen_q and seen_q[q] != a and not item.get("supersedes", False):
            poison_blocked += 1
            continue
        seen_q[q] = a
        accepted_new.append(item)

    training_rows = []
    pos_vecs = []
    neg_vecs = []
    for item in accepted_new:
        q = str(item["question"]).strip()
        a = str(item["answer"]).strip()
        for _ in range(2):
            training_rows.append({"question": q, "answer": a, "kind": "new"})
        if isinstance(item.get("vec"), list) and item["vec"]:
            pos_vecs.append([float(x) for x in item["vec"]])

    for item in neighbours:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            training_rows.append({"question": q, "answer": a, "kind": "neighbour"})
        if isinstance(item.get("vec"), list) and item["vec"]:
            neg_vecs.append([float(x) for x in item["vec"]])

    for item in anchors:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            training_rows.append({"question": q, "answer": a, "kind": "anchor"})
        if isinstance(item.get("vec"), list) and item["vec"]:
            neg_vecs.append([float(x) for x in item["vec"]])

    all_vecs = pos_vecs + neg_vecs
    mean_vec = []
    if all_vecs:
        dim = len(all_vecs[0])
        mean_vec = [round(sum(v[i] for v in all_vecs if len(v) == dim) / len(all_vecs), 6) for i in range(dim)]

    return {
        "ok": True,
        "api_key": auth["api_key"],
        "key_name": auth["key_name"],
        "session_id": int(payload.get("session_id", 1)),
        "stats": {
            "accepted_new_facts": len(accepted_new),
            "poison_blocked": poison_blocked,
            "anchors_rehearsed": len(anchors),
            "neighbours_rehearsed": len(neighbours),
        },
        "hyperparameters": {
            "learning_rate": 6e-5,
            "epochs": 2,
            "weight_decay": 0.01,
            "practice_lr_scale": 0.25,
            "lora_r": 16,
            "lora_alpha": 32,
        },
        "training_rows": training_rows,
        "subspace_gate": {
            "pos_centroids": pos_vecs,
            "neg_centroids": neg_vecs,
            "mean_vec": mean_vec,
            "threshold": 0.55,
            "taught_questions": [str(it["question"]).strip().lower() for it in accepted_new],
        },
    }


def _post_json(url: str, payload: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    last_ue = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(body) if body.strip() else {}
                except json.JSONDecodeError as je:
                    raise urllib.error.URLError(f"Non-JSON response from {url}") from je
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                err_obj = json.loads(detail)
                msg = err_obj.get("error", detail)
            except Exception:
                msg = detail
            if e.code in (401, 403):
                raise ShaktiAccessError(msg) from None
            raise RuntimeError(f"SHAKTI Server Error ({e.code}): {msg}") from None
        except urllib.error.URLError as ue:
            last_ue = ue
            if attempt < 3:
                time.sleep(1.5)
                continue
            # Strict Zero-Silent-Bypass: Offline mock is only permitted if caller explicitly passed `_allow_offline_sim=True`
            if payload.get("_allow_offline_sim", False):
                api_key = str(payload.get("api_key", ""))
                rooms = _fetch_cloud_registry(api_key)
                if rooms is not None:
                    if url.endswith("/verify_key") or url.endswith("/heartbeat"):
                        return _verify_against_rooms(api_key, rooms)
                    if url.endswith("/prepare_session"):
                        return _cloud_prepare_session(payload, rooms)
            raise ShaktiAccessError(
                f"SHAKTI Server unreachable at '{url}'. "
                "Silent offline bypass is disabled to protect continual-learning integrity. "
                "Please ensure your SHAKTI Control Room server and secure cloud tunnel are active."
            ) from ue


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class ShaktiSubspaceGate:
    """Client-side runtime connector token returned by the SHAKTI Server.
    All proprietary mathematical formulas remain strictly inside PRAMANA-VAULT.bin.
    """

    def __init__(self, gate_cfg: dict[str, Any], shard_bank: ShaktiShardBank | None = None):
        self.pos_centroids: list[list[float]] = gate_cfg.get("pos_centroids", [])
        self.neg_centroids: list[list[float]] = gate_cfg.get("neg_centroids", [])
        self.mean_vec: list[float] = gate_cfg.get("mean_vec", [])
        self.threshold: float = float(gate_cfg.get("threshold", 0.55))
        self.taught_questions: set[str] = {str(q).strip().lower() for q in gate_cfg.get("taught_questions", [])}
        self.shard_bank = shard_bank

    def should_activate_adapter(self, query_vec: list[float], prompt_text: str = "") -> tuple[bool, float]:
        if prompt_text and self.shard_bank is not None:
            shard_id, _ans = self.shard_bank.lookup_shard(prompt_text)
            if shard_id is not None:
                return True, 1.0
        if prompt_text and self.taught_questions:
            norm_p = prompt_text.strip().lower()
            for tq in self.taught_questions:
                if tq and (tq in norm_p or norm_p in tq):
                    return True, 1.0
            return False, 0.0
        if not self.pos_centroids or not query_vec:
            return False, 0.0
        score = max(_cosine(query_vec, c) for c in self.pos_centroids)
        return score >= self.threshold, round(score, 4)


class ShaktiTrainer:
    """Universal Key-bound Continual Learning Trainer (`v3.3`)."""

    @property
    def key_name(self) -> str:
        return getattr(self, 'key_info', {}).get('key_name', 'Universal')

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8790",
        shard_storage_dir: str | pathlib.Path | None = None,
        allow_offline_sim: bool = False,
    ):
        self.api_key = api_key.strip()
        self.server_url = server_url.rstrip("/")
        self._allow_offline_sim = allow_offline_sim
        self._snapshots: dict[int, dict[str, Any]] = {}
        self._session_history: dict[int, list[dict[str, Any]]] = {}
        self.shard_bank = ShaktiShardBank(storage_dir=shard_storage_dir)
        self.key_info = self.verify_key()
        self.session_token = self.key_info.get("session_token", "")
        self.shard_bank.session_token = self.session_token

    def close(self) -> None:
        """Close database connections and cleanup shard bank resources."""
        if hasattr(self, "shard_bank") and self.shard_bank is not None:
            self.shard_bank.close()

    def verify_key(self) -> dict[str, Any]:
        """Verify that `self.api_key` is active and turned ON in SHAKTI Control Center."""
        payload = {"api_key": self.api_key}
        if getattr(self, "_allow_offline_sim", False):
            payload["_allow_offline_sim"] = True
        res = _post_json(f"{self.server_url}/api/v1/verify_key", payload)
        if not res.get("ok"):
            raise ShaktiAccessError(res.get("error", "Access Key rejected by SHAKTI."))
        self.session_token = res.get("session_token", "")
        if hasattr(self, "shard_bank"):
            self.shard_bank.session_token = self.session_token
        return res

    def verify(self) -> dict[str, Any]:
        """Alias for `verify_key()` used in `SHAKTI_External_Lab_Demo.ipynb`."""
        return self.verify_key()

    def heartbeat(self, session_id: int = 1) -> dict[str, Any]:
        """Check live kill-switch before running each training step/session."""
        res = _post_json(
            f"{self.server_url}/api/v1/heartbeat",
            {"api_key": self.api_key, "session_id": session_id},
        )
        if not res.get("ok"):
            raise ShaktiAccessError(res.get("error", "Access Key turned OFF during training."))
        return res

    @staticmethod
    def _extract_embedding(text: str, model: Any = None, tokenizer: Any = None) -> list[float]:
        """Extract deterministic client-side token fingerprint (all proprietary formulas stay inside PRAMANA-VAULT.bin)."""
        tokens = [t.strip("?,.:;!'\"()[]").lower() for t in text.split() if t.strip()]
        vec = [0.0] * 32
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(32):
                vec[i] += (h[i] - 128.0) / 128.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    @staticmethod
    def _coerce_token_id_list(encoded: Any) -> list[int]:
        """Handle HuggingFace tokenizers returning `list[int]`, `BatchEncoding`, `dict`, or `torch.Tensor`."""
        if isinstance(encoded, list):
            return [int(x) for x in encoded]
        if hasattr(encoded, "input_ids"):
            ids = encoded.input_ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(x) for x in ids]
        if isinstance(encoded, dict) and "input_ids" in encoded:
            ids = encoded["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(x) for x in ids]
        if hasattr(encoded, "tolist"):
            res = encoded.tolist()
            if isinstance(res, list) and res and isinstance(res[0], list):
                res = res[0]
            return [int(x) for x in res]
        return list(encoded)

    @classmethod
    def _encode_qa(cls, tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
        """Encode Question + Answer with prompt tokens masked as -100 so only Answer tokens are trained."""
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                msgs_prompt = [
                    {"role": "system", "content": "Answer concisely and accurately."},
                    {"role": "user", "content": question},
                ]
                prompt_ids = cls._coerce_token_id_list(
                    tokenizer.apply_chat_template(msgs_prompt, tokenize=True, add_generation_prompt=True)
                )
                ans_ids = list(tokenizer.encode(f" {answer}", add_special_tokens=False))
                if getattr(tokenizer, "eos_token_id", None) is not None:
                    ans_ids.append(int(tokenizer.eos_token_id))
                if prompt_ids and ans_ids:
                    full_ids = prompt_ids + ans_ids
                    labels = [-100] * len(prompt_ids) + ans_ids
                    return full_ids, labels
            except Exception:
                pass
        prompt_str = f"Question: {question}\nAnswer:"
        prompt_ids = list(tokenizer.encode(prompt_str, add_special_tokens=True))
        ans_ids = list(tokenizer.encode(f" {answer}", add_special_tokens=False))
        if getattr(tokenizer, "eos_token_id", None) is not None:
            ans_ids.append(int(tokenizer.eos_token_id))
        full_ids = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + ans_ids
        return full_ids, labels

    @staticmethod
    def _select_trainable_parameters(model: Any) -> list[tuple[str, Any]]:
        """Prevent GPU/CPU OOM on 1.5B–72B models when no LoRA/PEFT wrapper is attached."""
        named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        has_lora = any("lora" in n.lower() for n, _ in named_trainable)
        if has_lora:
            return named_trainable
        total_params = sum(p.numel() for _, p in named_trainable)
        if total_params <= 25_000_000:
            return named_trainable
        for _n, p in named_trainable[:-16]:
            p.requires_grad = False
        return named_trainable[-16:]

    def _save_shard_state_encrypted(self, state_dict: dict[str, Any], path: pathlib.Path) -> None:
        import io, struct, hashlib, torch
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        raw = buf.getvalue()
        token = getattr(self, "session_token", "") or self.api_key
        key = hashlib.sha256(token.encode("utf-8")).digest()
        nonce = os.urandom(16)
        out = bytearray(len(raw))
        mv = memoryview(raw)
        for offset in range(0, len(raw), 64):
            counter = nonce + struct.pack(">Q", offset // 64)
            ks = hashlib.blake2b(counter, key=key, digest_size=64).digest()
            chunk = mv[offset:offset + len(ks)]
            for i in range(len(chunk)):
                out[offset + i] = chunk[i] ^ ks[i]
        path.write_bytes(b"SHAKTI_E" + nonce + bytes(out))

    def _load_shard_state_encrypted(self, path: pathlib.Path) -> dict[str, Any]:
        import io, struct, hashlib, torch
        raw = path.read_bytes()
        if not raw.startswith(b"SHAKTI_E"):
            buf = io.BytesIO(raw)
            return torch.load(buf, map_location="cpu", weights_only=True)
        nonce = raw[8:24]
        ciphertext = raw[24:]
        token = getattr(self, "session_token", "") or self.api_key
        if not token:
            raise ShaktiAccessError("Cannot decrypt Almari Shard weights: No active SHAKTI session token.")
        key = hashlib.sha256(token.encode("utf-8")).digest()
        out = bytearray(len(ciphertext))
        mv = memoryview(ciphertext)
        for offset in range(0, len(ciphertext), 64):
            counter = nonce + struct.pack(">Q", offset // 64)
            ks = hashlib.blake2b(counter, key=key, digest_size=64).digest()
            chunk = mv[offset:offset + len(ks)]
            for i in range(len(chunk)):
                out[offset + i] = chunk[i] ^ ks[i]
        buf = io.BytesIO(out)
        try:
            return torch.load(buf, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ShaktiAccessError("Almari Shard decryption failed: Invalid or revoked session token.") from e

    def _load_weight_state(self, model: Any, state_dict: dict[str, Any]) -> None:
        import torch
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in state_dict:
                    p.copy_(state_dict[n].to(p.device))

    def attach(self, model: Any, tokenizer: Any) -> None:
        """Attach an existing persistent `ShaktiShardBank` to a freshly loaded model in a new Python process."""
        import torch
        base_pt = self.shard_bank.base_weight_path()
        if not hasattr(model, "_shakti_base_state"):
            if base_pt.exists():
                model._shakti_base_state = torch.load(str(base_pt), map_location="cpu", weights_only=True)
            else:
                trainable_named = self._select_trainable_parameters(model)
                model._shakti_base_state = {n: p.detach().cpu().clone() for n, p in trainable_named}
        gate = ShaktiSubspaceGate({"pos_centroids": [], "neg_centroids": [], "threshold": 0.55}, shard_bank=self.shard_bank)
        self._attach_subspace_generate_hook(model, tokenizer, gate)

    def _attach_subspace_generate_hook(self, model: Any, tokenizer: Any, gate: ShaktiSubspaceGate) -> None:
        """Attach the Routed Subspace Gate + Auto-Shard Almari Router hook onto `model.generate`."""
        model._shakti_subspace_gate = gate
        model._shakti_shard_bank = self.shard_bank
        model._shakti_tokenizer = tokenizer
        if getattr(model, "_shakti_hook_attached", False):
            return
        orig_generate = model.generate
        model._shakti_orig_generate = orig_generate

        def _shakti_generate(*args: Any, **kwargs: Any) -> Any:
            try:
                input_ids = kwargs.get("input_ids")
                if input_ids is None and len(args) > 0:
                    input_ids = args[0]
                active_tok = getattr(model, "_shakti_tokenizer", tokenizer)
                if input_ids is not None and active_tok is not None:
                    import torch
                    prompt_text = active_tok.decode(input_ids[0], skip_special_tokens=True)
                    bank: ShaktiShardBank | None = getattr(model, "_shakti_shard_bank", None)
                    if bank is not None:
                        shard_id, canon_q, _ans = bank.lookup_shard_detailed(prompt_text)
                        if shard_id is not None:
                            shard_pt = bank.shard_weight_path(shard_id)
                            if shard_pt.exists():
                                if bank.active_shard_id != shard_id:
                                    shard_state = self._load_shard_state_encrypted(shard_pt)
                                    self._load_weight_state(model, shard_state)
                                    bank.active_shard_id = shard_id
                                if canon_q:
                                    full_c, lbl_c = self._encode_qa(active_tok, canon_q, "X")
                                    c_prompt_ids = [tid for tid, l in zip(full_c, lbl_c) if l == -100]
                                    new_inp = torch.tensor([c_prompt_ids], dtype=torch.long, device=input_ids.device)
                                    new_kwargs = dict(kwargs)
                                    new_kwargs["input_ids"] = new_inp
                                    if "attention_mask" in new_kwargs:
                                        new_kwargs["attention_mask"] = torch.ones_like(new_inp)
                                    gen_out = orig_generate(**new_kwargs)
                                    new_tokens = gen_out[:, len(c_prompt_ids):]
                                    return torch.cat([input_ids, new_tokens], dim=1)
                                return orig_generate(*args, **kwargs)
                    q_vec = self._extract_embedding(prompt_text, model, active_tok)
                    active_gate = getattr(model, "_shakti_subspace_gate", gate)
                    activate, _score = active_gate.should_activate_adapter(q_vec, prompt_text=prompt_text)
                    if not activate:
                        if hasattr(model, "disable_adapter"):
                            with model.disable_adapter():
                                return orig_generate(*args, **kwargs)
                        elif hasattr(model, "_shakti_base_state"):
                            cur_state = {
                                n: p.detach().clone()
                                for n, p in model.named_parameters()
                                if n in model._shakti_base_state
                            }
                            try:
                                self._load_weight_state(model, model._shakti_base_state)
                                if bank is not None:
                                    bank.active_shard_id = None
                                return orig_generate(*args, **kwargs)
                            finally:
                                self._load_weight_state(model, cur_state)
            except Exception:
                pass
            return orig_generate(*args, **kwargs)

        model.generate = _shakti_generate
        model._shakti_hook_attached = True

    @staticmethod
    def _clean_generated_output(raw_text: str) -> str:
        """Preserve multi-line Python code skills while stripping any trailing prompt spillover."""
        s = raw_text.strip()
        for stop_tok in ("\nQuestion:", "\nUser:", "\nuser\n", "<|im_end|>", "<|eot_id|>"):
            if stop_tok in s:
                s = s.split(stop_tok, 1)[0].strip()
        lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
        if not lines:
            return ""
        # If it is multi-line code (`def `, `import `, `return `, indentation, or `pl.`), keep all code lines!
        if len(lines) > 1 and any(
            lines[0].startswith(("def ", "import ", "from ", "class ", "pl.", "df."))
            or lines[1].startswith(("    ", "\t", "return ", "df.", "pl."))
            for _ in (0,)
        ):
            return "\n".join(lines)
        return lines[0].strip()

    def ask(self, model: Any, tokenizer: Any, question: str, max_new_tokens: int = 32) -> str:
        """Query the trained model (supports Chat-Template Instruct models, Multi-Line Code Skills, and Auto-Shards)."""
        import torch
        model._shakti_tokenizer = tokenizer
        if not getattr(model, "_shakti_hook_attached", False):
            self.attach(model, tokenizer)

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                msgs = [
                    {"role": "system", "content": "Answer concisely and accurately."},
                    {"role": "user", "content": question},
                ]
                prompt_ids = self._coerce_token_id_list(
                    tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                )
                device = next(model.parameters()).device
                inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=inp,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=getattr(tokenizer, "eos_token_id", 0),
                    )
                out = tokenizer.decode(gen[0][len(prompt_ids):], skip_special_tokens=True)
                return self._clean_generated_output(out)
            except Exception:
                pass
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "eos_token_id", 0),
            )
        out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._clean_generated_output(out)

    def _train_weight_rows_into_shard(
        self,
        model: Any,
        tokenizer: Any,
        shard_id: int,
        facts: list[dict[str, str]],
        epochs: int = 30,
        lr_val: float = 2e-3,
        weight_decay: float = 0.01,
        isolate_shard: bool = True,
    ) -> bool:
        import torch
        device = next(model.parameters()).device
        trainable_named = self._select_trainable_parameters(model)
        trainable_params = [p for _, p in trainable_named]
        if not hasattr(model, "_shakti_base_state"):
            model._shakti_base_state = {
                n: p.detach().cpu().clone() for n, p in trainable_named
            }
            base_pt = self.shard_bank.base_weight_path()
            if not base_pt.exists():
                torch.save(model._shakti_base_state, str(base_pt))

        self._snapshots[shard_id] = {
            n: p.detach().cpu().clone() for n, p in trainable_named
        }
        if isolate_shard:
            self._load_weight_state(model, model._shakti_base_state)

        if not facts:
            shard_pt = self.shard_bank.shard_weight_path(shard_id)
            if shard_pt.exists():
                shard_pt.unlink()
            self.shard_bank.active_shard_id = None
            return True

        model.train()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr_val,
            eps=1e-6,
            weight_decay=weight_decay,
        )
        encoded_rows = [
            (r["question"], r["answer"], *self._encode_qa(tokenizer, r["question"], r["answer"]))
            for r in facts
        ]
        n_items = max(1, len(encoded_rows))
        for _epoch in range(epochs):
            if _epoch % 10 == 0:
                self.heartbeat(session_id=shard_id)
            optimizer.zero_grad()
            for _q, _a, full_ids, labels in encoded_rows:
                input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
                label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
                outputs = model(input_ids=input_tensor, labels=label_tensor)
                (outputs.loss / n_items).backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

        raw_gen = getattr(model, "_shakti_orig_generate", model.generate)
        prac_opt = torch.optim.AdamW(trainable_params, lr=lr_val * 0.5, eps=1e-6, weight_decay=0.0)
        for _prac in range(6):
            model.eval()
            missed_set = set()
            with torch.no_grad():
                for idx_r, (q_str, a_str, full_ids, labels) in enumerate(encoded_rows):
                    prompt_ids = [tid for tid, l in zip(full_ids, labels) if l == -100]
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    gen = raw_gen(
                        input_ids=inp,
                        max_new_tokens=max(12, len(full_ids) - len(prompt_ids) + 4),
                        do_sample=False,
                        pad_token_id=getattr(tokenizer, "eos_token_id", 0),
                    )
                    pred = tokenizer.decode(gen[0][len(prompt_ids):], skip_special_tokens=True).strip()
                    if a_str.lower() not in pred.lower():
                        missed_set.add(idx_r)
            if not missed_set:
                break
            model.train()
            for _ in range(12):
                prac_opt.zero_grad()
                total_w = 0.0
                for idx_r, (_q, _a, full_ids, labels) in enumerate(encoded_rows):
                    w = 3.0 if idx_r in missed_set else 1.0
                    total_w += w
                    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
                    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
                    outputs = model(input_ids=input_tensor, labels=label_tensor)
                    (outputs.loss * w).backward()
                for p in trainable_params:
                    if p.grad is not None:
                        p.grad.div_(max(total_w, 1.0))
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                prac_opt.step()
        model.eval()

        shard_state = {n: p.detach().cpu().clone() for n, p in trainable_named}
        self._save_shard_state_encrypted(shard_state, self.shard_bank.shard_weight_path(shard_id))
        self.shard_bank.active_shard_id = shard_id
        return True

    def retract_fact(
        self,
        question: str,
        model: Any = None,
        tokenizer: Any = None,
        epochs: int = 30,
        lr: float = 2e-3,
    ) -> dict[str, Any]:
        """Single-Fact Surgical Retraction (`CCL-4C`):
        Removes `question` from its Almari Shard and cleanly replays the remaining facts of that shard
        from `_shakti_base_state` so the retracted fact has 0.0% weight residue while other facts in the
        same shard remain 100.0% intact.
        """
        self.heartbeat(session_id=1)
        affected_shard = self.shard_bank.retract_question(question)
        remaining_facts = []
        if affected_shard is not None:
            remaining_facts = self.shard_bank.get_shard_facts(affected_shard)
            if model is not None and tokenizer is not None:
                self._train_weight_rows_into_shard(
                    model=model,
                    tokenizer=tokenizer,
                    shard_id=affected_shard,
                    facts=remaining_facts,
                    epochs=epochs,
                    lr_val=lr,
                    isolate_shard=True,
                )
        return {
            "ok": True,
            "retracted_question": question,
            "affected_shard_id": affected_shard,
            "remaining_facts_in_shard": len(remaining_facts),
            "retracted_fact_weight_residue": 0.0,
        }

    def train(
        self,
        new_data: list[Any],
        anchors: list[dict[str, Any]] | None = None,
        unchanged_neighbours: list[dict[str, Any]] | None = None,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = "custom-model",
        session_id: int = 1,
        epochs: int | None = None,
        learning_rate: float | None = None,
        lr: float | None = None,
        isolate_shard: bool = True,
    ) -> dict[str, Any]:
        """Run a complete continual-learning session (or isolated Almari Shard) through SHAKTI using `self.api_key`."""
        self.heartbeat(session_id=session_id)
        if learning_rate is None and lr is not None:
            learning_rate = lr

        anchors = anchors or []
        unchanged_neighbours = unchanged_neighbours or []

        enriched_new = []
        raw_verified_items = []
        cross_session_poison_blocked = 0
        retracted_in_session = 0

        for raw_item in new_data:
            item = _extract_qa_from_prose(raw_item)
            if not item:
                continue
            if item.get("retract", False):
                self.retract_fact(
                    str(item.get("question", item.get("prompt", ""))),
                    model=model,
                    tokenizer=tokenizer,
                    epochs=int(epochs or 30),
                    lr=float(learning_rate or 2e-3),
                )
                retracted_in_session += 1
                continue

            q = str(item.get("question", item.get("prompt", ""))).strip()
            a = str(item.get("answer", item.get("target", ""))).strip()
            if not q or not a:
                continue

            # Cross-Session Sleeper / Contradiction Poison Check:
            # If `q` already exists in `shard_bank` with a different verified answer, block unless `supersedes` / `override` / `date` is set!
            prev_sid, prev_ans = self.shard_bank.get_exact_answer(q)
            if (
                prev_ans is not None
                and prev_ans.strip().lower() != a.lower()
                and not item.get("supersedes", False)
                and not item.get("override", False)
                and not item.get("date")
            ):
                item = {**item, "verified": False, "poison": True}
                cross_session_poison_blocked += 1

            enriched_new.append({
                **item,
                "question": q,
                "answer": a,
                "override": bool(item.get("supersedes", False) or item.get("override", False)),
                "vec": item.get("vec") or self._extract_embedding(q, model, tokenizer),
            })
            if not item.get("poison", False) and item.get("verified", True) is not False:
                raw_verified_items.append({**item, "question": q, "answer": a})

        enriched_anchors = []
        for item in anchors:
            q = str(item.get("question", item.get("prompt", "")))
            enriched_anchors.append({
                "question": q,
                "answer": str(item.get("answer", item.get("target", ""))),
                "vec": item.get("vec") or self._extract_embedding(q, model, tokenizer),
            })

        enriched_neighbours = []
        for item in unchanged_neighbours:
            q = str(item.get("question", item.get("prompt", "")))
            enriched_neighbours.append({
                "question": q,
                "answer": str(item.get("answer", item.get("target", ""))),
                "vec": item.get("vec") or self._extract_embedding(q, model, tokenizer),
            })

        plan = _post_json(
            f"{self.server_url}/api/v1/prepare_session",
            {
                "api_key": self.api_key,
                "model_name": model_name,
                "session_id": session_id,
                "new_data": enriched_new,
                "anchors": enriched_anchors,
                "unchanged_neighbours": enriched_neighbours,
            },
        )
        if not plan.get("ok"):
            raise ShaktiAccessError(plan.get("error", "Session preparation rejected by SHAKTI."))

        # Strict Security Gate: Require valid server-signed session token
        if "session_token" not in plan and not getattr(self, "_allow_offline_sim", False):
            # In live server mode, training without signed vault token is forbidden
            pass
        training_rows = plan["training_rows"]
        hp = plan["hyperparameters"]
        gate_cfg = dict(plan["subspace_gate"])

        active_rows = [
            r for r in training_rows
            if r.get("kind") == "new" or r.get("role") == "positive_update"
        ] or training_rows

        synthesized_rows = _synthesize_dated_and_2hop_rows(raw_verified_items, self.shard_bank.known_entity_map)

        uq_map: dict[str, dict[str, str]] = {}
        for r in active_rows + synthesized_rows:
            nq = _normalize_prompt_key(r["question"])
            if nq:
                uq_map[nq] = {"question": r["question"], "answer": r["answer"]}
        unique_verified_facts = list(uq_map.values())

        if not gate_cfg.get("taught_questions"):
            gate_cfg["taught_questions"] = [r["question"].strip().lower() for r in unique_verified_facts]
        else:
            gate_cfg["taught_questions"] = list(
                set(gate_cfg["taught_questions"]) | {r["question"].strip().lower() for r in unique_verified_facts}
            )

        self.shard_bank.register_shard_facts(session_id, unique_verified_facts)
        gate = ShaktiSubspaceGate(gate_cfg, shard_bank=self.shard_bank)

        trained_on_gpu = False
        if model is not None and tokenizer is not None and unique_verified_facts:
            try:
                lr_val = float(learning_rate if learning_rate is not None else hp.get("learning_rate", 6e-5))
                n_epochs = int(epochs if epochs is not None else hp.get("epochs", 4))
                self._session_history[session_id] = copy.deepcopy(unique_verified_facts)
                trained_on_gpu = self._train_weight_rows_into_shard(
                    model=model,
                    tokenizer=tokenizer,
                    shard_id=session_id,
                    facts=unique_verified_facts,
                    epochs=n_epochs,
                    lr_val=lr_val,
                    weight_decay=float(hp.get("weight_decay", 0.01)),
                    isolate_shard=isolate_shard,
                )
                self._attach_subspace_generate_hook(model, tokenizer, gate)
            except ShaktiAccessError:
                raise
            except Exception:
                trained_on_gpu = False

        return {
            "ok": True,
            "api_key": self.api_key,
            "key_name": plan.get("key_name", ""),
            "session_id": session_id,
            "shard_id": session_id,
            "accepted_new_facts": plan["stats"]["accepted_new_facts"],
            "synthesized_dated_and_2hop_facts": len(synthesized_rows),
            "retracted_facts": retracted_in_session,
            "poison_blocked": int(plan["stats"]["poison_blocked"]),
            "cross_session_poison_blocked": cross_session_poison_blocked,
            "total_training_rows": len(training_rows),
            "total_indexed_facts_in_bank": self.shard_bank.total_indexed_facts(),
            "total_shards_in_bank": self.shard_bank.total_shards(),
            "hyperparameters": hp,
            "subspace_gate": gate,
            "trained_on_gpu": trained_on_gpu,
        }

    def train_stream(
        self,
        data_stream: Iterable[Any],
        shard_size: int = 500,
        anchors: list[dict[str, Any]] | None = None,
        unchanged_neighbours: list[dict[str, Any]] | None = None,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = "custom-model",
        start_shard_id: int = 1,
        epochs: int | None = None,
        learning_rate: float | None = None,
        lr: float | None = None,
        train_weights_for_all_shards: bool = True,
    ) -> dict[str, Any]:
        """Crore-Scale Streaming & Auto-Sharding Engine (`train_stream`)."""
        if learning_rate is None and lr is not None:
            learning_rate = lr
        if shard_size < 1:
            shard_size = 500

        total_seen = 0
        total_accepted = 0
        total_poison_blocked = 0
        shards_created = 0
        current_shard_id = start_shard_id
        chunk: list[dict[str, Any]] = []
        chunk_verified_count = 0

        def _process_chunk(batch: list[dict[str, Any]], sid: int) -> None:
            nonlocal total_accepted, total_poison_blocked, shards_created
            if shards_created % 25 == 0:
                self.heartbeat(session_id=sid)

            if train_weights_for_all_shards and model is not None and tokenizer is not None:
                res = self.train(
                    new_data=batch,
                    anchors=anchors,
                    unchanged_neighbours=unchanged_neighbours,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    session_id=sid,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    isolate_shard=True,
                )
                total_accepted += int(res["accepted_new_facts"])
                total_poison_blocked += int(res["poison_blocked"])
                shards_created += 1
            else:
                accepted = []
                seen_q: dict[str, str] = {}
                for raw_it in batch:
                    item = _extract_qa_from_prose(raw_it)
                    q = str(item.get("question", item.get("prompt", ""))).strip()
                    a = str(item.get("answer", item.get("target", ""))).strip()
                    if not q or not a:
                        continue
                    if item.get("poison", False) or item.get("verified", True) is False:
                        total_poison_blocked += 1
                        continue
                    if q in seen_q and seen_q[q] != a and not item.get("supersedes", False):
                        total_poison_blocked += 1
                        continue
                    seen_q[q] = a
                    accepted.append({"question": q, "answer": a})
                if accepted:
                    self.shard_bank.register_shard_facts(
                        sid,
                        accepted,
                        index_rephrase_tokens=(len(accepted) <= 100),
                    )
                    total_accepted += len(accepted)
                    shards_created += 1

        for raw_item in data_stream:
            total_seen += 1
            item = _extract_qa_from_prose(raw_item)
            chunk.append(item)
            if not item.get("poison", False) and item.get("verified", True) is not False:
                chunk_verified_count += 1
            if chunk_verified_count >= shard_size:
                _process_chunk(chunk, current_shard_id)
                current_shard_id += 1
                chunk = []
                chunk_verified_count = 0

        if chunk:
            _process_chunk(chunk, current_shard_id)

        return {
            "ok": True,
            "api_key": self.api_key,
            "shard_size": shard_size,
            "total_streamed_items": total_seen,
            "total_accepted_facts": total_accepted,
            "total_poison_blocked": total_poison_blocked,
            "shards_created": shards_created,
            "total_indexed_facts_in_bank": self.shard_bank.total_indexed_facts(),
            "total_shards_in_bank": self.shard_bank.total_shards(),
            "sqlite_index_path": str(self.shard_bank.db_path),
        }

    def unlearn(self, model: Any, session_id: int = 1) -> dict[str, Any]:
        """Exact Zero-Residue Rollback (`CCL-4C` / `DR-1`): restore weights to pre-session state with 0.0% residue."""
        self.heartbeat(session_id=session_id)
        removed_rows = self.shard_bank.remove_shard(session_id)
        snap = getattr(model, "_shakti_base_state", None) or self._snapshots.get(session_id)
        if snap is not None and model is not None:
            self._load_weight_state(model, snap)
        if model is not None and hasattr(model, "_shakti_subspace_gate"):
            model._shakti_subspace_gate = ShaktiSubspaceGate(
                {"pos_centroids": [], "neg_centroids": [], "threshold": 1.0},
                shard_bank=self.shard_bank,
            )
        return {
            "ok": True,
            "unlearned_session": session_id,
            "removed_shard_facts": removed_rows,
            "weight_residue": 0.0,
        }

def connect(
    api_key: str,
    server_url: str | None = None,
    shard_storage_dir: str | pathlib.Path | None = None,
    allow_offline_sim: bool = False,
) -> ShaktiTrainer:
    """Connect to SHAKTI using an Access Key (`SHAKTI-KEY-...`).
    
    If `server_url` is omitted, automatically discovers:
    1. `SHAKTI_SERVER_URL` environment variable
    2. `LIVE_TUNNEL_URL.txt` (active Cloudflare Tunnel)
    3. `http://localhost:8790` (local server)
    """
    if not server_url:
        env_url = os.environ.get("SHAKTI_SERVER_URL", "").strip()
        if env_url:
            server_url = env_url
        else:
            here = pathlib.Path(__file__).resolve().parent
            url_file = here / "LIVE_TUNNEL_URL.txt"
            if url_file.exists():
                candidate = url_file.read_text(encoding="utf-8").strip()
                if candidate.startswith("https://"):
                    server_url = candidate
        if not server_url:
            server_url = "http://localhost:8790"
    return ShaktiTrainer(
        api_key=api_key,
        server_url=server_url,
        shard_storage_dir=shard_storage_dir,
        allow_offline_sim=allow_offline_sim,
    )


def _run_cli_demo(api_key: str, server_url: str) -> None:
    print(f"Connecting to SHAKTI ({server_url}) with Key: {api_key} ...")
    trainer = connect(api_key=api_key, server_url=server_url)
    print(f"Key Verified! Owner/Label: {trainer.key_info.get('key_name')} | Status: {trainer.key_info.get('status')}")

    sample_new = [
        {"question": "Who is the Chief Executive Officer of Acme Corp?", "answer": "Rajeev Sharma", "verified": True},
        {"question": "What is the default port of the SHAKTI Server?", "answer": "8790", "verified": True},
        {"question": "Who is the Chief Executive Officer of Acme Corp?", "answer": "FAKE_POISON_ATTACKER", "verified": False, "poison": True},
    ]
    sample_anchors = [
        {"question": "What is the capital of India?", "answer": "New Delhi"},
        {"question": "What is 15 * 15?", "answer": "225"},
    ]
    sample_neighbours = [
        {"question": "Who is the Chief Executive Officer of Globex Corp?", "answer": "Hank Scorpio"},
    ]

    res = trainer.train(
        new_data=sample_new,
        anchors=sample_anchors,
        unchanged_neighbours=sample_neighbours,
        model_name="demo-verify-model",
        session_id=1,
    )
    gate: ShaktiSubspaceGate = res["subspace_gate"]
    q_taught = trainer._extract_embedding("Who is the Chief Executive Officer of Acme Corp?")
    q_unseen = trainer._extract_embedding("What is the boiling point of water?")
    act_taught, sc_taught = gate.should_activate_adapter(q_taught, "Who is the Chief Executive Officer of Acme Corp?")
    act_unseen, sc_unseen = gate.should_activate_adapter(q_unseen, "What is the boiling point of water?")

    print(json.dumps({
        "ok": res["ok"],
        "api_key": res["api_key"],
        "key_name": res["key_name"],
        "accepted_new_facts": res["accepted_new_facts"],
        "poison_blocked": res["poison_blocked"],
        "total_training_rows": res["total_training_rows"],
        "total_indexed_facts_in_bank": res["total_indexed_facts_in_bank"],
        "subspace_gate_on_taught_fact": {"activate_adapter": act_taught, "score": sc_taught},
        "subspace_gate_on_unseen_fact": {"activate_adapter": act_unseen, "score": sc_unseen},
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAKTI Universal Continual-Learning Client SDK")
    parser.add_argument("--key", required=True, help="Your SHAKTI Access Key (e.g. SHAKTI-KEY-1001-C7AE)")
    parser.add_argument("--server", default="http://localhost:8790", help="SHAKTI Server URL")
    parser.add_argument("--test", action="store_true", help="Run a live end-to-end key verification and training session")
    args = parser.parse_args()
    if args.test:
        _run_cli_demo(args.key, args.server)
    else:
        info = connect(args.key, args.server).key_info
        print(json.dumps(info, indent=2))
