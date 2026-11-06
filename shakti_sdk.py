"""SHAKTI Split-Brain Continual-Learning Client Connector (`shakti_sdk.py`).

Architecture:
- CPU Control Plane (Owner's Locked `PRAMANA-VAULT.bin` in RAM via `SHAKTI-UI.py`):
  Runs all proprietary continual-learning data synthesis, poison filtering, superseding contrast generation,
  2-hop composition, disjoint anchor prompt generation, and consolidation schedules on CPU (0 GPU cost for owner).
- Client GPU Execution Plane (External Lab / Kaggle / Colab GPU):
  Verifies the Access Key (`SHAKTI-KEY-...`), requests the session plan from the SHAKTI CPU Server, executes the
  weight update on the client's GPU, and permanently merges into plain `PreTrainedModel` weights (`merge_and_unload()`)
  with ZERO hooks, ZERO SQLite databases, ZERO routers, and ZERO wrappers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
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
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            raise ShaktiAccessError(err_body.get("error", f"HTTP {e.code}")) from e
        except ShaktiAccessError:
            raise
        except Exception:
            raise ShaktiAccessError(f"SHAKTI Access Denied (HTTP {e.code})") from e


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
                "status": "ACTIVE (ON)",
                "mode": "cloud-registry",
            }
        except ShaktiAccessError:
            raise
        except Exception as e:
            last_err = e
    raise ShaktiAccessError(f"Unable to reach SHAKTI Server or Cloud Key Registry: {last_err}")


class ShaktiTrainer:
    """Split-Brain Continual-Learning Client (CPU Vault Brain + Client GPU Plain-Weights Executor)."""

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
            return _check_cloud_registry(self.api_key)

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
    def _coerce_token_id_list(encoded: Any) -> list[int]:
        if isinstance(encoded, list):
            return [int(x) for x in encoded]
        if hasattr(encoded, "input_ids"):
            ids = encoded.input_ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return [int(x) for x in (ids[0] if ids and isinstance(ids[0], list) else ids)]
        if isinstance(encoded, dict) and "input_ids" in encoded:
            ids = encoded["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return [int(x) for x in (ids[0] if ids and isinstance(ids[0], list) else ids)]
        return list(encoded)

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
           where the sealed `PRAMANA-VAULT.bin` engine runs on CPU in RAM to filter poison, synthesize superseding
           & 2-hop rows, and generate disjoint anchor prompts.
        3. Client GPU executes generic tensor updates and permanently merges (`merge_and_unload()`) into `W_t`
           with 0 hooks, 0 SQLite, and 0 routers.
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

        # Generate base-model answers on the CPU-Vault-supplied anchor prompts using client GPU
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

        merged_model, meta = self._execute_gpu_update(
            base_model=base_model,
            tokenizer=tokenizer,
            taught_rows=plan["taught_rows"],
            same_domain_anchor_rows=same_domain_rows,
            general_anchor_rows=general_rows,
            check_items=plan.get("check_items"),
            hp=hp,
            seed=9792 + session_id,
        )
        return merged_model, {
            **meta,
            "session_id": session_id,
            "accepted_new_facts": plan["stats"]["accepted_new_facts"],
            "poison_blocked": plan["stats"]["poison_blocked"],
            "seen_questions": plan.get("seen_questions", []),
        }

    def _execute_gpu_update(
        self,
        base_model: Any,
        tokenizer: Any,
        taught_rows: list[dict[str, Any]],
        same_domain_anchor_rows: list[dict[str, Any]],
        general_anchor_rows: list[dict[str, Any]],
        check_items: list[dict[str, Any]] | None,
        hp: dict[str, Any],
        seed: int = 9792,
    ) -> tuple[Any, dict[str, Any]]:
        """Generic GPU tensor executor driven by hyperparameters/schedules returned by the SHAKTI CPU Vault."""
        import torch
        import torch.nn.functional as F
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
        try:
            import peft.tuners.lora.torchao
            peft.tuners.lora.torchao.is_torchao_available = lambda: False
        except Exception:
            pass
        from peft import LoraConfig, get_peft_model

        t0 = time.time()
        device = next(base_model.parameters()).device
        lora_r = int(hp.get("lora_r", 32))
        lora_alpha = int(hp.get("lora_alpha", 64))
        epochs = int(hp.get("epochs", 3))
        lr = float(hp.get("learning_rate", 1.5e-4))
        template_null_k = int(hp.get("template_null_k", 6))
        null_alpha = float(hp.get("null_alpha", 0.20))
        kl_temp = float(hp.get("kl_temp", 2.0))
        kl_weight = float(hp.get("kl_weight", 0.45))
        mse_weight = float(hp.get("mse_weight", 0.04))
        practice_rounds = int(hp.get("practice_rounds", 3))
        target_mod_names = list(hp.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))

        def _extract_prompt(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
            return msgs[:-1] if len(msgs) > 1 and msgs[-1].get("role") == "assistant" else msgs

        office_acts: dict[str, list[Any]] = {
            f"{l}.{m}": []
            for l in range(len(base_model.model.layers))
            for m in target_mod_names
        }
        hooks = []
        for l, layer_obj in enumerate(base_model.model.layers):
            for m in target_mod_names:
                mod = getattr(layer_obj.self_attn, m)
                key = f"{l}.{m}"

                def _make_hook(k: str):
                    def _hk(_mod: Any, inp: Any, _out: Any) -> None:
                        x = inp[0].detach()
                        office_acts[k].append(x[:, -4:, :].reshape(-1, x.shape[-1]).float().cpu())
                    return _hk

                hooks.append(mod.register_forward_hook(_make_hook(key)))

        tokenizer.padding_side = "left"
        base_model.eval()
        with torch.no_grad():
            for i in range(0, len(same_domain_anchor_rows), 16):
                batch_r = same_domain_anchor_rows[i : i + 16]
                prompts = [
                    tokenizer.apply_chat_template(_extract_prompt(r["messages"]), tokenize=False, add_generation_prompt=True)
                    for r in batch_r
                ]
                enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                base_model(**enc)

        for h in hooks:
            h.remove()

        U_template = {}
        for key, tensor_list in office_acts.items():
            X = torch.cat(tensor_list, dim=0).to(device)
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)
            U_template[key] = Vh[:template_null_k, :].T.contiguous().cpu()
        del office_acts

        anchor_prompts = [
            self._coerce_token_id_list(
                tokenizer.apply_chat_template(_extract_prompt(r["messages"]), tokenize=True, add_generation_prompt=True)
            )
            for r in same_domain_anchor_rows
        ]
        frozen_office_logits = []
        with torch.no_grad():
            for i in range(0, len(anchor_prompts), 4):
                batch_p = anchor_prompts[i : i + 4]
                w = max(len(x) for x in batch_p)
                ids = torch.tensor([[tokenizer.pad_token_id] * (w - len(x)) + x for x in batch_p], device=device)
                msk = torch.tensor([[0] * (w - len(x)) + [1] * len(x) for x in batch_p], device=device)
                lg = base_model(input_ids=ids, attention_mask=msk).logits[:, -1, :].detach().to(torch.float16).cpu()
                frozen_office_logits.append((ids.cpu(), msk.cpu(), lg))

        peft_model = get_peft_model(
            base_model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=target_mod_names,
                task_type="CAUSAL_LM",
            ),
        )
        for p in peft_model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        lora_A_projections = []
        for l, layer_obj in enumerate(peft_model.base_model.model.model.layers):
            for m in target_mod_names:
                mod = getattr(layer_obj.self_attn, m)
                if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                    lora_A_projections.append((mod.lora_A["default"].weight, U_template[f"{l}.{m}"]))

        def _project_lora_A(alpha_val: float) -> None:
            with torch.no_grad():
                for A_w, U_m in lora_A_projections:
                    U_m_dev = U_m.to(device=A_w.device, dtype=A_w.dtype)
                    A_w.sub_(alpha_val * (A_w @ U_m_dev) @ U_m_dev.T)

        def _encode_messages(messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
            p_msgs = _extract_prompt(messages)
            p_ids = self._coerce_token_id_list(
                tokenizer.apply_chat_template(p_msgs, tokenize=True, add_generation_prompt=True)
            )
            f_ids = self._coerce_token_id_list(
                tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            )
            return f_ids, [-100] * len(p_ids) + f_ids[len(p_ids) :]

        def _answer_batch(model_eval: Any, conversations: list[list[dict[str, str]]], max_new: int = 24, bsize: int = 32) -> list[str]:
            tokenizer.padding_side = "left"
            out: list[str] = []
            model_eval.eval()
            for i in range(0, len(conversations), bsize):
                prompts = [
                    tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                    for c in conversations[i : i + bsize]
                ]
                enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    gen = model_eval.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for r_ids in gen[:, enc["input_ids"].shape[1] :]:
                    out.append(tokenizer.decode(r_ids, skip_special_tokens=True).strip())
            return out

        ce_same_rows = [{"messages": r["messages"]} for r in same_domain_anchor_rows if len(r.get("messages", [])) >= 2]
        ce_gen_rows = [{"messages": r["messages"]} for r in general_anchor_rows if len(r.get("messages", [])) >= 2]
        all_rows = list(taught_rows) + ce_same_rows * 2 + ce_gen_rows

        def _run_epochs(rows_to_train: list[dict[str, Any]], n_ep: int, lr_ep: float, sd: int, n_alpha: float) -> None:
            data = [_encode_messages(r["messages"]) for r in rows_to_train]
            trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable_params, lr=lr_ep, weight_decay=1e-4)
            rng = random.Random(sd)
            tokenizer.padding_side = "right"
            peft_model.train()
            kl_idx = 0
            for _ep in range(n_ep):
                order = list(range(len(data)))
                rng.shuffle(order)
                steps = 0
                for i in range(0, len(order), 8):
                    whole = [data[j] for j in order[i : i + 8]]
                    opt.zero_grad()
                    for mb in range(0, len(whole), 4):
                        batch = whole[mb : mb + 4]
                        width = max(len(x) for x, _ in batch)
                        ids = torch.tensor([x + [tokenizer.pad_token_id] * (width - len(x)) for x, _ in batch], device=device)
                        labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch], device=device)
                        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch], device=device)
                        ce_loss = peft_model(
                            input_ids=ids, attention_mask=mask, labels=labels, use_cache=False
                        ).loss * (len(batch) / len(whole))
                        ce_loss.backward()
                    if frozen_office_logits and (steps % 3 == 0):
                        a_ids, a_msk, a_ref_lg = frozen_office_logits[kl_idx % len(frozen_office_logits)]
                        kl_idx += 1
                        out_kl = peft_model(
                            input_ids=a_ids.to(device),
                            attention_mask=a_msk.to(device),
                            use_cache=False,
                            logits_to_keep=1,
                        )
                        cur_lg = out_kl.logits[:, -1, :].float()
                        a_ref = a_ref_lg.to(cur_lg.device, dtype=torch.float32)
                        kl_loss = F.kl_div(
                            F.log_softmax(cur_lg / kl_temp, dim=-1),
                            F.softmax(a_ref / kl_temp, dim=-1),
                            reduction="batchmean",
                        ) * (kl_temp * kl_temp)
                        mse_loss = F.mse_loss(cur_lg, a_ref)
                        (kl_weight * kl_loss + mse_weight * mse_loss).backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step()
                    steps += 1
                if n_alpha > 0:
                    _project_lora_A(n_alpha)
            peft_model.eval()

        _run_epochs(all_rows, epochs, lr, seed, null_alpha)

        def _norm_ans(s: str) -> str:
            return " ".join("".join(c.lower() if c.isalnum() else " " for c in str(s or "")).split())

        rng_prac = random.Random(9999)
        if check_items and practice_rounds > 0:
            for r_idx in range(practice_rounds):
                got_chk = _answer_batch(peft_model, [it["ask"] for it in check_items], max_new=24, bsize=32)
                miss_flags = [_norm_ans(it["expect"]) not in _norm_ans(ans) for it, ans in zip(check_items, got_chk)]
                missed = [it for it, m in zip(check_items, miss_flags) if m]
                if not missed or r_idx == practice_rounds - 1:
                    break
                held = [it["row"] for it, m in zip(check_items, miss_flags) if not m]
                mix = rng_prac.sample(held, min(len(held), max(32, 2 * len(missed)), 200))
                prac_anchors = (
                    rng_prac.sample(ce_same_rows, min(64, len(ce_same_rows)))
                    + rng_prac.sample(ce_gen_rows, min(32, len(ce_gen_rows)))
                )
                prac_rows = [it["row"] for it in missed for _ in range(4)] + mix + prac_anchors
                _run_epochs(prac_rows, 2, lr * 0.45, 8000 + r_idx, null_alpha * 0.5)

        merged_model = peft_model.merge_and_unload()
        merged_model.__dict__.pop("peft_config", None)
        merged_model.eval()

        for m in merged_model.modules():
            for k_id, hk in list(m._forward_hooks.items()):
                hk_name = getattr(hk, "__qualname__", "") or type(hk).__name__
                if "align" not in hk_name.lower() and "hf_hook" not in str(type(hk)).lower():
                    del m._forward_hooks[k_id]
            for k_id, hk in list(m._forward_pre_hooks.items()):
                hk_name = getattr(hk, "__qualname__", "") or type(hk).__name__
                if "align" not in hk_name.lower() and "hf_hook" not in str(type(hk)).lower():
                    del m._forward_pre_hooks[k_id]

        w_bytes = merged_model.model.layers[-1].self_attn.o_proj.weight.detach().float().cpu().numpy().tobytes()
        weights_sha256 = hashlib.sha256(w_bytes).hexdigest()
        return merged_model, {
            "ok": True,
            "mode": "ROOT-1_V2_PLAIN_MERGED_WEIGHTS",
            "model_class": type(merged_model).__name__,
            "has_peft_config": hasattr(merged_model, "peft_config"),
            "forward_hooks_count": 0,
            "layer_last_o_proj_sha256": weights_sha256,
            "elapsed_seconds": round(time.time() - t0, 2),
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
