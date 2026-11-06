"""SHAKTI Split-Brain Continual-Learning Client Connector (`shakti_sdk.py` v3.3 — NASA/DeepMind Grade).

Architecture & Guarantees:
- CPU Control Plane (Owner's Locked `PRAMANA-VAULT.bin` in RAM via `SHAKTI-UI.py`):
  Runs 100% of proprietary continual-learning data synthesis (`TX-1 v2`), `Badha v2` poison filtering,
  `Pratipaksa-Bhavana` superseding contrast generation, O(1) Entity-Graph 2-Hop/3-Hop composition, and disjoint
  `Svabhava-Anchor` prompt selection on the owner's CPU (0 GPU cost for owner).
- Client GPU Execution Plane (External Lab / Kaggle / Colab / Cloud GPU):
  Contains ZERO formula code on disk or GitHub. Authenticates the Access Key (`SHAKTI-KEY-...`), discovers the live
  encrypted tunnel via 0-TTL GitHub API, executes the ephemeral update on the client's GPU with automatic failover,
  and permanently merges into plain `PreTrainedModel` weights (`merge_and_unload()`) with ZERO hooks, ZERO SQLite,
  and ZERO wrappers.
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import time
import urllib.error
import urllib.request
import zlib
from typing import Any

# Primary 0-TTL GitHub REST API endpoint (bypasses 300s Fastly CDN cache) + Raw fallbacks
CLOUD_REGISTRY_URLS = (
    "https://api.github.com/repos/MEMGATE/SHAKTI-CLOUD-KEYS/contents/company-rooms.json",
    "https://api.github.com/repos/MEMGATE/SHAKTI-CLOUD-KEYS/contents/handoff/company-rooms.json",
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI-CLOUD-KEYS/main/company-rooms.json",
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI-CLOUD-KEYS/main/handoff/company-rooms.json",
)


class ShaktiAccessError(PermissionError):
    """Raised when a SHAKTI Access Key is OFF, deleted, or invalid."""


def _safe_apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tokenize: bool = False,
    add_generation_prompt: bool = True,
) -> Any:
    """Universal chat-template helper compatible with Qwen, Llama-3, Mistral, Gemma-2, Phi-3, and base models."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            merged_msgs: list[dict[str, str]] = []
            sys_buf: list[str] = []
            for m in messages:
                role = m.get("role", "user")
                content = str(m.get("content", "")).strip()
                if role == "system":
                    if content:
                        sys_buf.append(content)
                elif role == "user":
                    if sys_buf:
                        content = "\n\n".join(sys_buf + [content])
                        sys_buf.clear()
                    merged_msgs.append({"role": "user", "content": content})
                else:
                    merged_msgs.append({"role": role, "content": content})
            try:
                return tokenizer.apply_chat_template(
                    merged_msgs,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

    parts: list[str] = []
    for m in messages:
        r = m.get("role", "user")
        c = str(m.get("content", "")).strip()
        if r == "system":
            parts.append(f"System: {c}")
        elif r == "user":
            parts.append(f"Question: {c}")
        elif r == "assistant":
            parts.append(f"Answer: {c}")
    if add_generation_prompt:
        parts.append("Answer:")
    text_prompt = "\n".join(parts)
    if tokenize:
        return tokenizer.encode(text_prompt, add_special_tokens=True)
    return text_prompt


def _decrypt_cloud_tunnel_url(api_key: str, rm_entry: dict[str, Any]) -> str:
    """Decrypt `enc_server_url` using `api_key` (or fall back to plaintext `server_url` if present)."""
    enc_hex = str(rm_entry.get("enc_server_url", "")).strip()
    if enc_hex:
        try:
            raw = bytes.fromhex(enc_hex)
            if len(raw) >= 16:
                nonce, ct = raw[:16], raw[16:]
                k_enc = hashlib.sha256(f"SHAKTI-URL-KEY:{api_key}".encode("utf-8")).digest()
                ks = hashlib.blake2b(nonce, key=k_enc, digest_size=64).digest()
                pt = bytes(b ^ ks[i % 64] for i, b in enumerate(ct)).decode("utf-8").strip().rstrip("/")
                if pt:
                    return pt
        except Exception:
            pass
    return str(rm_entry.get("server_url", "")).strip().rstrip("/")


def _post_json(url: str, payload: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SHAKTI-SDK/3.3",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}"
        try:
            err_body = json.loads(e.read().decode("utf-8", errors="replace"))
            if isinstance(err_body, dict) and err_body.get("error"):
                err_msg = str(err_body["error"])
        except Exception:
            pass
        if e.code in (401, 403):
            raise ShaktiAccessError(err_msg) from e
        raise RuntimeError(f"SHAKTI Server {err_msg} (HTTP {e.code})") from e


def _check_cloud_registry(api_key: str) -> dict[str, Any]:
    """Query the 0-TTL GitHub API & fallback registries, supporting both zero-knowledge hashed keys and plaintext keys."""
    key_hash_id = "sha256:" + hashlib.sha256(f"SHAKTI-REG-ID:{api_key}".encode("utf-8")).hexdigest()
    last_err: Exception | None = None
    saw_valid_registry = False

    for reg_url in CLOUD_REGISTRY_URLS:
        try:
            is_api = "api.github.com" in reg_url
            headers = {
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "SHAKTI-SDK/3.3",
            }
            if is_api:
                headers["Accept"] = "application/vnd.github.raw+json"
                fetch_url = reg_url
            else:
                fetch_url = f"{reg_url}?t={int(time.time())}"

            req = urllib.request.Request(fetch_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                rooms = json.loads(resp.read().decode("utf-8"))
            if not isinstance(rooms, dict):
                continue
            saw_valid_registry = True

            rm = rooms.get(key_hash_id) or rooms.get(api_key)
            if not isinstance(rm, dict):
                continue

            if not rm.get("switch_on", False):
                raise ShaktiAccessError(
                    f"BLOCKED BY SHAKTI: Access Key '{api_key}' is turned OFF in SHAKTI Control Center."
                )
            srv_url = _decrypt_cloud_tunnel_url(api_key, rm)
            return {
                "ok": True,
                "api_key": api_key,
                "key_name": rm.get("company", "Verified Client"),
                "server_url": srv_url,
                "status": "ACTIVE (ON)",
                "mode": rm.get("mode", "UNLIMITED"),
            }
        except ShaktiAccessError:
            raise
        except Exception as e:
            last_err = e

    if saw_valid_registry:
        raise ShaktiAccessError(f"BLOCKED BY SHAKTI: Access Key '{api_key}' does not exist or was deleted.")
    raise ShaktiAccessError(f"Unable to reach SHAKTI Server or Cloud Key Registry: {last_err}")


class ShaktiTrainer:
    """Split-Brain Continual-Learning Client Connector (0 Formula Code on Disk — Powered by Owner's RAM Vault)."""

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://localhost:8790",
        trajectory_id: str = "default",
        verbose: bool = True,
    ):
        self.api_key = str(api_key or "").strip()
        if not self.api_key.startswith("SHAKTI-KEY-"):
            raise ShaktiAccessError(f"Invalid SHAKTI Access Key format: '{self.api_key}'")
        self.server_url = str(server_url or "http://localhost:8790").rstrip("/")
        self.trajectory_id = str(trajectory_id or "default").strip() or "default"
        self.verbose = bool(verbose)
        self._model_sig: tuple[Any, ...] | None = None
        self._same_domain_rows: list[dict[str, Any]] = []
        self._general_rows: list[dict[str, Any]] = []
        self._cached_u_template: dict[str, Any] | None = None
        self._cached_frozen_logits: list[Any] | None = None
        self._seen_questions: list[str] = []
        self._session_seq: int = 0
        self.key_info = self.verify_key()

    def reset_session(self, new_trajectory_id: str | None = None) -> None:
        """Reset client-side continual learning caches and start a clean trajectory without restarting the Vault."""
        self._same_domain_rows = []
        self._general_rows = []
        self._cached_u_template = None
        self._cached_frozen_logits = None
        self._seen_questions = []
        self._session_seq = 0
        if new_trajectory_id:
            self.trajectory_id = str(new_trajectory_id).strip()
        try:
            _post_json(
                f"{self.server_url}/api/v1/reset_session",
                {"api_key": self.api_key, "trajectory_id": self.trajectory_id},
                timeout=6.0,
            )
        except Exception:
            pass

    def _request_with_failover(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float = 35.0,
        max_retries: int = 6,
    ) -> dict[str, Any]:
        """Execute POST request to the SHAKTI Vault Server with automatic 0-TTL tunnel re-discovery and exponential
        backoff (`1.5s -> 3s -> 5s -> 8s -> 12s -> 18s`) if the reverse tunnel rotates mid-session.
        """
        backoffs = (1.5, 3.0, 5.0, 8.0, 12.0, 18.0)
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                res = _post_json(f"{self.server_url}{endpoint}", payload, timeout=timeout)
                if not res.get("ok"):
                    raise ShaktiAccessError(res.get("error", f"Request to {endpoint} rejected by SHAKTI Vault."))
                return res
            except ShaktiAccessError:
                raise
            except Exception as e:
                last_exc = e
                # Immediately check 0-TTL cloud registry for a newly rotated tunnel URL
                try:
                    cloud_info = _check_cloud_registry(self.api_key)
                    new_srv = cloud_info.get("server_url", "")
                    if new_srv and new_srv != self.server_url:
                        if self.verbose:
                            print(f"[SHAKTI] 🔄 Tunnel rotated -> hot-swapping to {new_srv}", flush=True)
                        self.server_url = new_srv
                        # Try the newly discovered tunnel URL immediately before sleeping
                        try:
                            res = _post_json(f"{self.server_url}{endpoint}", payload, timeout=timeout)
                            if not res.get("ok"):
                                raise ShaktiAccessError(res.get("error", f"Request to {endpoint} rejected."))
                            return res
                        except ShaktiAccessError:
                            raise
                        except Exception as e2:
                            last_exc = e2
                except ShaktiAccessError:
                    raise
                except Exception:
                    pass

                if attempt < max_retries - 1:
                    wait_s = backoffs[min(attempt, len(backoffs) - 1)]
                    time.sleep(wait_s)

        raise RuntimeError(f"Failed to reach SHAKTI Vault Server at {self.server_url}{endpoint}: {last_exc}")

    def verify_key(self) -> dict[str, Any]:
        try:
            res = _post_json(f"{self.server_url}/api/v1/verify_key", {"api_key": self.api_key}, timeout=6.0)
            if not res.get("ok"):
                raise ShaktiAccessError(res.get("error", "Access Key rejected."))
            res.setdefault("server_url", self.server_url)
            return res
        except ShaktiAccessError:
            raise
        except Exception:
            cloud_info = _check_cloud_registry(self.api_key)
            tunnel_url = cloud_info.get("server_url", "")
            if tunnel_url and tunnel_url != self.server_url:
                self.server_url = tunnel_url
                try:
                    res = _post_json(f"{self.server_url}/api/v1/verify_key", {"api_key": self.api_key}, timeout=8.0)
                    if res.get("ok"):
                        res.setdefault("server_url", self.server_url)
                        return res
                except ShaktiAccessError:
                    raise
                except Exception:
                    pass
            return cloud_info

    def heartbeat(self, session_id: int = 1) -> dict[str, Any]:
        try:
            res = _post_json(
                f"{self.server_url}/api/v1/heartbeat",
                {"api_key": self.api_key, "session_id": session_id},
                timeout=6.0,
            )
            if not res.get("ok"):
                raise ShaktiAccessError(res.get("error", "Heartbeat rejected."))
            return res
        except ShaktiAccessError:
            raise
        except Exception:
            cloud_info = _check_cloud_registry(self.api_key)
            tunnel_url = cloud_info.get("server_url", "")
            if tunnel_url and tunnel_url != self.server_url:
                self.server_url = tunnel_url
            return cloud_info

    @staticmethod
    def ask_plain_weights_batch(
        model: Any,
        tokenizer: Any,
        questions: list[str],
        max_new_tokens: int = 24,
        batch_size: int = 16,
    ) -> list[str]:
        """Batched greedy probing of a plain merged model (`batch_size=16`, >15x faster than sequential probing)."""
        import torch

        if not questions:
            return []
        device = next(model.parameters()).device
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        orig_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        model.eval()
        out_answers: list[str] = []
        try:
            for i in range(0, len(questions), batch_size):
                batch_q = questions[i : i + batch_size]
                prompts = [
                    _safe_apply_chat_template(
                        tokenizer,
                        [
                            {"role": "system", "content": "Answer concisely and accurately with the exact name or value."},
                            {"role": "user", "content": q},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for q in batch_q
                ]
                enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for r_ids in gen[:, enc["input_ids"].shape[1] :]:
                    ans = tokenizer.decode(r_ids, skip_special_tokens=True).strip()
                    out_answers.append(ans.splitlines()[0].strip() if ans else "")
        finally:
            tokenizer.padding_side = orig_padding_side
        return out_answers

    @staticmethod
    def ask_plain_weights(model: Any, tokenizer: Any, question: str, max_new_tokens: int = 24) -> str:
        """Query a plain merged model directly via `model.generate` (0 hooks, 0 adapters)."""
        res = ShaktiTrainer.ask_plain_weights_batch(
            model, tokenizer, [question], max_new_tokens=max_new_tokens, batch_size=1
        )
        return res[0] if res else ""

    def train_session_via_vault(
        self,
        base_model: Any,
        tokenizer: Any,
        new_items: list[Any],
        prior_questions: list[str] | None = None,
        session_id: int | None = None,
        reset_session: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """True Split-Brain Continual Learning:
        1. Automatically tracks `session_id` and `prior_questions` if omitted by caller, and probes `W_{t-1}` using
           fast batched GPU generation (`batch_size=16`, bounded to O(1) rehearsal budget).
        2. Sends `new_items` + live weight predictions to the SHAKTI CPU Vault Server (`/api/v1/prepare_session`)
           with 6-attempt automatic tunnel failover.
        3. Decrypts the one-time session-locked GPU kernel in RAM, updates the client's GPU model into plain
           merged weights (`W_t`), and immediately wipes the ephemeral kernel from RAM.
        """
        t_start = time.time()
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Detect model architecture changes and auto-invalidate cached tensors if model changed
        cfg = getattr(base_model, "config", None)
        cur_sig = (
            getattr(cfg, "_name_or_path", type(base_model).__name__),
            getattr(cfg, "hidden_size", 0),
            getattr(cfg, "vocab_size", 0),
        )
        if self._model_sig is not None and self._model_sig != cur_sig:
            self.reset_session()
        self._model_sig = cur_sig

        if reset_session or (session_id == 1 and prior_questions is not None and len(prior_questions) == 0):
            self._same_domain_rows = []
            self._general_rows = []
            self._cached_u_template = None
            self._cached_frozen_logits = None
            self._seen_questions = []
            self._session_seq = 0

        if session_id is None:
            self._session_seq += 1
            eff_session_id = self._session_seq
        else:
            eff_session_id = int(session_id)
            self._session_seq = max(self._session_seq, eff_session_id)

        eff_prior_questions = list(prior_questions) if prior_questions is not None else list(self._seen_questions)
        if len(eff_prior_questions) > 128:
            # Bounded O(1) age-stratified rehearsal probe
            recent_q = eff_prior_questions[-64:]
            older_pool = eff_prior_questions[:-64]
            step = max(1, len(older_pool) // 64)
            eff_prior_questions = older_pool[::step][:64] + recent_q

        self.heartbeat(session_id=eff_session_id)

        weight_predictions: dict[str, str] = {}
        if eff_prior_questions:
            batch_ans = self.ask_plain_weights_batch(
                base_model, tokenizer, eff_prior_questions, max_new_tokens=24, batch_size=16
            )
            for q_prev, ans in zip(eff_prior_questions, batch_ans):
                if ans:
                    weight_predictions[q_prev] = ans

        req_payload = {
            "api_key": self.api_key,
            "trajectory_id": self.trajectory_id,
            "reset_session": bool(reset_session),
            "session_id": eff_session_id,
            "new_data": new_items,
            "weight_predictions": weight_predictions,
        }
        plan = self._request_with_failover("/api/v1/prepare_session", req_payload, timeout=40.0, max_retries=6)

        enc_kernel_b64 = plan.get("encrypted_gpu_kernel", "")
        session_token = plan.get("session_token", "")
        if not enc_kernel_b64 or not session_token:
            raise ShaktiAccessError("SHAKTI Vault Server did not return an authenticated GPU session kernel.")

        same_domain_prompts = plan.get("same_domain_anchor_prompts", [])
        general_prompts = plan.get("general_anchor_prompts", [])
        hp = plan.get("hyperparameters", {})

        def _fill_anchor_prompts(prompt_list: list[list[dict[str, str]]], max_new: int) -> list[dict[str, Any]]:
            import torch
            device = next(base_model.parameters()).device
            orig_pad_side = getattr(tokenizer, "padding_side", "right")
            tokenizer.padding_side = "left"
            out_r: list[dict[str, Any]] = []
            base_model.eval()
            try:
                for i in range(0, len(prompt_list), 16):
                    bc = prompt_list[i : i + 16]
                    pr = [_safe_apply_chat_template(tokenizer, c, tokenize=False, add_generation_prompt=True) for c in bc]
                    enc = tokenizer(pr, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                    with torch.no_grad():
                        gn = base_model.generate(
                            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id
                        )
                    for c, r_ids in zip(bc, gn[:, enc["input_ids"].shape[1] :]):
                        raw_ans = tokenizer.decode(r_ids, skip_special_tokens=True).strip()
                        ans = raw_ans.splitlines()[0].strip() if raw_ans else ""
                        if ans:
                            out_r.append({"messages": c + [{"role": "assistant", "content": ans}]})
            finally:
                tokenizer.padding_side = orig_pad_side
            return out_r

        if not self._same_domain_rows:
            self._same_domain_rows = _fill_anchor_prompts(same_domain_prompts, 20)
            self._general_rows = _fill_anchor_prompts(general_prompts, 32)

        # Decrypt one-time session-locked GPU kernel in ephemeral RAM, execute, and immediately wipe
        enc_bytes = base64.b64decode(enc_kernel_b64)
        stream_key = hashlib.sha256(f"{self.api_key}:{session_token}".encode("utf-8")).digest()
        raw_comp = bytes(
            b ^ hashlib.sha256(stream_key + idx.to_bytes(4, "big")).digest()[idx % 32]
            for idx, b in enumerate(enc_bytes)
        )
        kernel_src = zlib.decompress(raw_comp).decode("utf-8")
        del enc_bytes, raw_comp

        ephemeral_ns: dict[str, Any] = {"__name__": "shakti_ephemeral_gpu_session"}
        try:
            exec(kernel_src, ephemeral_ns)
            del kernel_src
            train_fn = ephemeral_ns["train_and_merge_root1_v2"]
            call_kwargs: dict[str, Any] = {
                "base_model": base_model,
                "tokenizer": tokenizer,
                "taught_rows": plan["taught_rows"],
                "same_domain_anchor_rows": self._same_domain_rows,
                "general_anchor_rows": self._general_rows,
                "check_items": plan.get("check_items"),
                "cached_u_template": self._cached_u_template,
                "cached_frozen_logits": self._cached_frozen_logits,
                "lora_r": int(hp.get("lora_r", 32)),
                "lora_alpha": int(hp.get("lora_alpha", 64)),
                "epochs": int(hp.get("epochs", 3)),
                "lr": float(hp.get("learning_rate", 1.5e-4)),
                "template_null_k": int(hp.get("template_null_k", 6)),
                "null_alpha": float(hp.get("null_alpha", 0.20)),
                "practice_rounds": int(hp.get("practice_rounds", 3)),
                "seed": 9792 + eff_session_id,
            }
            import inspect
            sig_params = inspect.signature(train_fn).parameters
            if "target_modules" in sig_params and hp.get("target_modules"):
                call_kwargs["target_modules"] = hp.get("target_modules")
            for _hp_k in ("kl_temp", "kl_weight", "mse_weight"):
                if _hp_k in sig_params and hp.get(_hp_k) is not None:
                    call_kwargs[_hp_k] = float(hp[_hp_k])
            merged_model, meta = train_fn(**call_kwargs)
            self._cached_u_template = meta.pop("_cached_u_template", self._cached_u_template)
            self._cached_frozen_logits = meta.pop("_cached_frozen_logits", self._cached_frozen_logits)
        finally:
            ephemeral_ns.clear()
            gc.collect()

        srv_seen = plan.get("seen_questions", [])
        if srv_seen:
            self._seen_questions = list(srv_seen)

        if self.verbose:
            print(
                f"[SHAKTI] ✅ Session {eff_session_id} merged into plain weights "
                f"(+{plan['stats']['accepted_new_facts']} facts, {plan['stats']['poison_blocked']} poison blocked, "
                f"{round(time.time() - t_start, 1)}s)",
                flush=True,
            )

        return merged_model, {
            **meta,
            "session_id": eff_session_id,
            "accepted_new_facts": plan["stats"]["accepted_new_facts"],
            "poison_blocked": plan["stats"]["poison_blocked"],
            "seen_questions": list(self._seen_questions),
        }

    # Alias for clean developer ergonomics
    learn = train_session_via_vault


def connect(
    api_key: str,
    server_url: str = "http://localhost:8790",
    trajectory_id: str = "default",
    verbose: bool = True,
) -> ShaktiTrainer:
    """Connect to SHAKTI using an Access Key (`SHAKTI-KEY-...`)."""
    return ShaktiTrainer(api_key=api_key, server_url=server_url, trajectory_id=trajectory_id, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAKTI Split-Brain Continual-Learning Client Connector")
    parser.add_argument("--key", required=True, help="Your SHAKTI Access Key (e.g. SHAKTI-KEY-1001-0DF1)")
    parser.add_argument("--server", default="http://localhost:8790", help="SHAKTI CPU Vault Server URL")
    args = parser.parse_args()
    info = connect(args.key, args.server).key_info
    print(json.dumps(info, indent=2))
