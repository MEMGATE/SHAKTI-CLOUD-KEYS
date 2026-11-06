"""SHAKTI Split-Brain Continual-Learning Client Connector (`shakti_sdk.py`).

Architecture:
- CPU Control Plane (Owner's Locked `PRAMANA-VAULT.bin` in RAM via `SHAKTI-UI.py`):
  Runs 100% of proprietary continual-learning data synthesis, poison filtering, superseding contrast generation,
  2-hop composition, and disjoint anchor prompt generation on the owner's CPU (0 GPU cost for owner), and streams
  an ephemeral session-encrypted execution payload only when an ACTIVE `SHAKTI-KEY-...` is authenticated.
- Client GPU Execution Plane (External Lab / Kaggle / Colab GPU):
  Contains ZERO formula code on disk or GitHub. Verifies the Access Key (`SHAKTI-KEY-...`), requests the session
  plan from the SHAKTI CPU Server, executes the ephemeral update on the client's GPU, and permanently merges into
  plain `PreTrainedModel` weights (`merge_and_unload()`) with ZERO hooks, ZERO SQLite, and ZERO wrappers.
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

CLOUD_REGISTRY_URLS = (
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI-CLOUD-KEYS/main/company-rooms.json",
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI-CLOUD-KEYS/main/handoff/company-rooms.json",
    "https://raw.githubusercontent.com/MEMGATE/SHAKTI/main/handoff/company-rooms.json",
)


class ShaktiAccessError(PermissionError):
    """Raised when a SHAKTI Access Key is OFF, deleted, or invalid."""


def _post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                raise ShaktiAccessError(err_body.get("error", f"HTTP {e.code}")) from e
            except ShaktiAccessError:
                raise
            except Exception:
                raise ShaktiAccessError(f"SHAKTI Access Denied (HTTP {e.code})") from e
        raise RuntimeError(f"SHAKTI Server HTTP {e.code}") from e


def _check_cloud_registry(api_key: str) -> dict[str, Any]:
    last_err: Exception | None = None
    for reg_url in CLOUD_REGISTRY_URLS:
        try:
            req = urllib.request.Request(
                f"{reg_url}?t={int(time.time())}",
                headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                rooms = json.loads(resp.read().decode("utf-8"))
            if not isinstance(rooms, dict):
                continue
            if api_key not in rooms:
                raise ShaktiAccessError(f"BLOCKED BY SHAKTI: Access Key '{api_key}' does not exist or was deleted.")
            rm = rooms[api_key]
            if not rm.get("switch_on", False):
                raise ShaktiAccessError(f"BLOCKED BY SHAKTI: Access Key '{api_key}' is turned OFF in SHAKTI Control Center.")
            return {
                "ok": True,
                "api_key": api_key,
                "key_name": rm.get("company", ""),
                "server_url": str(rm.get("server_url", "")).strip().rstrip("/"),
                "status": "ACTIVE (ON)",
                "mode": "cloud-registry",
            }
        except ShaktiAccessError:
            raise
        except Exception as e:
            last_err = e
    raise ShaktiAccessError(f"Unable to reach SHAKTI Server or Cloud Key Registry: {last_err}")


class ShaktiTrainer:
    """Thin Split-Brain Client Connector (0 Formula Code on Disk — Powered by Owner's RAM Vault)."""

    def __init__(self, api_key: str, server_url: str = "http://localhost:8790"):
        self.api_key = str(api_key or "").strip()
        if not self.api_key.startswith("SHAKTI-KEY-"):
            raise ShaktiAccessError(f"Invalid SHAKTI Access Key format: '{self.api_key}'")
        self.server_url = str(server_url or "http://localhost:8790").rstrip("/")
        self.key_info = self.verify_key()

    def verify_key(self) -> dict[str, Any]:
        try:
            res = _post_json(f"{self.server_url}/api/v1/verify_key", {"api_key": self.api_key}, timeout=4.0)
            if not res.get("ok"):
                raise ShaktiAccessError(res.get("error", "Access Key rejected."))
            return res
        except ShaktiAccessError:
            raise
        except Exception:
            cloud_info = _check_cloud_registry(self.api_key)
            tunnel_url = cloud_info.get("server_url", "")
            if tunnel_url and tunnel_url != self.server_url:
                self.server_url = tunnel_url
                try:
                    res = _post_json(f"{self.server_url}/api/v1/verify_key", {"api_key": self.api_key}, timeout=6.0)
                    if res.get("ok"):
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
                timeout=4.0,
            )
            if not res.get("ok"):
                raise ShaktiAccessError(res.get("error", "Heartbeat rejected."))
            return res
        except ShaktiAccessError:
            raise
        except Exception:
            return _check_cloud_registry(self.api_key)

    @staticmethod
    def ask_plain_weights(model: Any, tokenizer: Any, question: str, max_new_tokens: int = 24) -> str:
        """Query a plain merged model directly via `model.generate` (0 hooks, 0 adapters)."""
        import torch

        device = next(model.parameters()).device
        msgs = [
            {"role": "system", "content": "Answer concisely and accurately with the exact name or value."},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        ans = tokenizer.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return ans.splitlines()[0].strip() if ans else ""

    def train_session_via_vault(
        self,
        base_model: Any,
        tokenizer: Any,
        new_items: list[Any],
        prior_questions: list[str] | None = None,
        session_id: int = 1,
    ) -> tuple[Any, dict[str, Any]]:
        """True Split-Brain Continual Learning:
        1. Queries current model weights `W_{t-1}` on `prior_questions` (0 answer dictionaries stored).
        2. Sends `new_items` + live weight predictions to the SHAKTI CPU Vault Server (`/api/v1/prepare_session`),
           where the sealed `PRAMANA-VAULT.bin` engine runs on the owner's CPU in RAM.
        3. Decrypts the one-time session-locked GPU kernel in RAM, updates the client's GPU model into plain
           merged weights (`W_t`), and immediately wipes the ephemeral kernel from RAM.
        """
        self.heartbeat(session_id=session_id)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        weight_predictions: dict[str, str] = {}
        for q_prev in (prior_questions or []):
            ans = self.ask_plain_weights(base_model, tokenizer, q_prev, max_new_tokens=24)
            if ans:
                weight_predictions[q_prev] = ans

        plan = _post_json(
            f"{self.server_url}/api/v1/prepare_session",
            {
                "api_key": self.api_key,
                "session_id": session_id,
                "new_data": new_items,
                "weight_predictions": weight_predictions,
            },
            timeout=30.0,
        )
        if not plan.get("ok"):
            raise ShaktiAccessError(plan.get("error", "Session preparation rejected by SHAKTI Vault Server."))

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
            tokenizer.padding_side = "left"
            out_r: list[dict[str, Any]] = []
            base_model.eval()
            for i in range(0, len(prompt_list), 16):
                bc = prompt_list[i : i + 16]
                pr = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in bc]
                enc = tokenizer(pr, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    gn = base_model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                for c, r_ids in zip(bc, gn[:, enc["input_ids"].shape[1] :]):
                    ans = tokenizer.decode(r_ids, skip_special_tokens=True).strip().splitlines()[0].strip()
                    if ans:
                        out_r.append({"messages": c + [{"role": "assistant", "content": ans}]})
            return out_r

        same_domain_rows = _fill_anchor_prompts(same_domain_prompts, 20)
        general_rows = _fill_anchor_prompts(general_prompts, 32)

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
            merged_model, meta = train_fn(
                base_model=base_model,
                tokenizer=tokenizer,
                taught_rows=plan["taught_rows"],
                same_domain_anchor_rows=same_domain_rows,
                general_anchor_rows=general_rows,
                check_items=plan.get("check_items"),
                lora_r=int(hp.get("lora_r", 32)),
                lora_alpha=int(hp.get("lora_alpha", 64)),
                epochs=int(hp.get("epochs", 3)),
                lr=float(hp.get("learning_rate", 1.5e-4)),
                template_null_k=int(hp.get("template_null_k", 6)),
                null_alpha=float(hp.get("null_alpha", 0.20)),
                kl_temp=float(hp.get("kl_temp", 2.0)),
                kl_weight=float(hp.get("kl_weight", 0.45)),
                mse_weight=float(hp.get("mse_weight", 0.04)),
                practice_rounds=int(hp.get("practice_rounds", 3)),
                seed=9792 + session_id,
            )
        finally:
            ephemeral_ns.clear()
            gc.collect()

        return merged_model, {
            **meta,
            "session_id": session_id,
            "accepted_new_facts": plan["stats"]["accepted_new_facts"],
            "poison_blocked": plan["stats"]["poison_blocked"],
            "seen_questions": plan.get("seen_questions", []),
        }


def connect(api_key: str, server_url: str = "http://localhost:8790") -> ShaktiTrainer:
    """Connect to SHAKTI using an Access Key (`SHAKTI-KEY-...`)."""
    return ShaktiTrainer(api_key=api_key, server_url=server_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAKTI Split-Brain Continual-Learning Client Connector")
    parser.add_argument("--key", required=True, help="Your SHAKTI Access Key (e.g. SHAKTI-KEY-1001-E4E3)")
    parser.add_argument("--server", default="http://localhost:8790", help="SHAKTI CPU Vault Server URL")
    args = parser.parse_args()
    info = connect(args.key, args.server).key_info
    print(json.dumps(info, indent=2))
