"""SHAKTI Universal Thin Client SDK (shakti_sdk.py) — v4.0 (Zero-Leakage Thin Client Edition).
Pure HTTP connector to the sovereign SHAKTI Vault Server.

Zero Local Storage:
- No local SQLite database or disk-based cache.
- No client-side prompt-rewriting regex tricks.
- No proprietary continual learning algorithms or math formulas inside the SDK.

All multi-tenant isolation, memory indexing, zero-label semantic poison filtering,
temporal state revision tracking, and unlearning logic execute strictly inside
the sovereign SHAKTI Vault Server running on the host machine.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional

DEFAULT_VAULT_URL = os.environ.get("SHAKTI_VAULT_URL", "http://127.0.0.1:8790").rstrip("/")
DEFAULT_API_KEY = os.environ.get("SHAKTI_API_KEY", "SHAKTI-KEY-1002-B25D")


class ShaktiAccessError(PermissionError):
    """Raised when access to SHAKTI Vault is denied, key is OFF, or server is unreachable."""


def _normalize_prompt_key(text: str) -> str:
    """Normalize any question or chat-templated prompt (`Qwen`, `Llama`, `Mistral`, `Gemma`, or `Question: ...`)
    into a canonical lookup key.
    """
    s = str(text or "").strip()
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


class ShaktiClient:
    """Ultra-Thin Python Client for SHAKTI Sovereign Vault Server."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        vault_url: Optional[str] = None,
        timeout: float = 15.0,
        verify_on_init: bool = True,
    ):
        self.api_key = str(api_key or DEFAULT_API_KEY).strip()
        self.vault_url = str(vault_url or DEFAULT_VAULT_URL).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ShaktiAccessError("SHAKTI Access Key is required. Set SHAKTI_API_KEY or pass api_key=...")
        if verify_on_init:
            self.verify()

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.vault_url}{endpoint}"
        body = json.dumps({**payload, "api_key": self.api_key}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "ShaktiClient-v4-Thin",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_body = json.loads(e.read().decode("utf-8", errors="replace"))
                err_msg = err_body.get("error", "")
            except Exception:
                err_msg = str(e)
            raise ShaktiAccessError(f"SHAKTI Vault Error ({e.code}): {err_msg}") from e
        except urllib.error.URLError as e:
            raise ShaktiAccessError(
                f"Cannot connect to SHAKTI Vault Server at {self.vault_url}: {e.reason}. "
                "Ensure SHAKTI-UI is running and local port 8790 is active."
            ) from e

    def verify(self) -> dict[str, Any]:
        """Verify the Access Key against the Vault Server and retrieve active tenant status."""
        res = self._post("/api/v1/verify_key", {})
        if not res.get("ok"):
            raise ShaktiAccessError(res.get("error", "Key verification failed."))
        return res

    def heartbeat(self) -> dict[str, Any]:
        """Send a lightweight ping to verify that the Vault connection is alive."""
        return self._post("/api/v1/heartbeat", {})

    def learn(self, session_id: int, stream: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        """Stream new session facts into the sovereign Vault Server.

        The Vault Server performs:
        1. Zero-label semantic poison filtering (blocking trojans, injection, backdoors).
        2. Automatic temporal state revision resolution (tracking versioned updates).
        3. Multi-tenant cryptographic memory indexing.
        """
        payload = {
            "session_id": int(session_id),
            "data": list(stream),
        }
        res = self._post("/api/v1/learn", payload)
        if not res.get("ok"):
            raise ShaktiAccessError(res.get("error", "Failed to ingest session data into Vault."))
        return res

    def route_query(self, query: str) -> dict[str, Any]:
        """Query the sovereign Vault Server for verified state and contextual routing."""
        res = self._post("/api/v1/route_query", {"query": str(query).strip()})
        if not res.get("ok"):
            raise ShaktiAccessError(res.get("error", "Vault query routing failed."))
        return res

    def unlearn(self, target: str, session_id: int = 0) -> dict[str, Any]:
        """Trigger an unlearning directive on the sovereign Vault Server for a target entity or function."""
        return self._post("/api/v1/unlearn", {"target": str(target).strip(), "session_id": int(session_id)})

    def status(self) -> dict[str, Any]:
        """Retrieve the current memory bank status, active facts count, and blocked poison statistics."""
        return self._post("/api/v1/vault_status", {})

    def ask(
        self,
        query: str,
        model: Any = None,
        tokenizer: Any = None,
        max_new_tokens: int = 64,
        strict_verified: bool = False,
    ) -> str:
        """Query the system for an answer.

        1. Calls the sovereign Vault Server via route_query().
        2. If Vault access is revoked / key is OFF, immediately raises ShaktiAccessError.
        3. If Vault has verified state for this query:
           - If a neural model/tokenizer is provided, passes the verified state to the model for grounded generation.
           - Otherwise returns the verified canonical state string directly.
        4. If Vault does not have this fact:
           - If strict_verified is True, returns 'UNKNOWN_FACT'.
           - If a model is provided, generates directly from base model weights without vault guidance.
           - Otherwise returns 'UNKNOWN_FACT'.
        """
        route = self.route_query(query)
        found = route.get("found", False)
        ans = route.get("answer")

        if found and ans is not None:
            if model is not None and tokenizer is not None:
                return self._generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    guidance=str(ans),
                    max_new_tokens=max_new_tokens,
                )
            return str(ans)

        # Fact not in vault
        if strict_verified or (model is None or tokenizer is None):
            return "UNKNOWN_FACT"

        # Generate strictly from base weights
        return self._generate_with_model(
            model=model,
            tokenizer=tokenizer,
            query=query,
            guidance=None,
            max_new_tokens=max_new_tokens,
        )

    def _generate_with_model(
        self,
        model: Any,
        tokenizer: Any,
        query: str,
        guidance: Optional[str],
        max_new_tokens: int = 64,
    ) -> str:
        """Helper to invoke local neural model with or without verified vault state."""
        import torch

        if guidance is not None:
            prompt = (
                f"<|im_start|>system\nYou are an accurate AI assistant. Answer using verified state.<|im_end|>\n"
                f"<|im_start|>user\n{query}\n[Verified State: {guidance}]<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>system\nYou are an accurate AI assistant.<|im_end|>\n"
                f"<|im_start|>user\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        clean = raw.split("\n")[0].strip()
        return clean


# Backwards compatibility helper functions
_DEFAULT_CLIENT: Optional[ShaktiClient] = None


def get_default_client(api_key: Optional[str] = None, vault_url: Optional[str] = None) -> ShaktiClient:
    global _DEFAULT_CLIENT
    if (
        _DEFAULT_CLIENT is None
        or (api_key and _DEFAULT_CLIENT.api_key != api_key)
        or (vault_url and _DEFAULT_CLIENT.vault_url != vault_url)
    ):
        _DEFAULT_CLIENT = ShaktiClient(api_key=api_key, vault_url=vault_url)
    return _DEFAULT_CLIENT


def shakti_ingest(
    session_id: int,
    stream: list[Any],
    api_key: Optional[str] = None,
    vault_url: Optional[str] = None,
) -> dict[str, Any]:
    """Ingest a sequential stream of facts into the Vault Server."""
    client = get_default_client(api_key=api_key, vault_url=vault_url)
    return client.learn(session_id, stream)


def shakti_query(
    query: str,
    api_key: Optional[str] = None,
    vault_url: Optional[str] = None,
    model: Any = None,
    tokenizer: Any = None,
) -> str:
    """Query the system for verified state."""
    client = get_default_client(api_key=api_key, vault_url=vault_url)
    return client.ask(query, model=model, tokenizer=tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAKTI Thin Client SDK CLI")
    parser.add_argument("--key", default=DEFAULT_API_KEY, help="SHAKTI Access Key")
    parser.add_argument("--url", default=DEFAULT_VAULT_URL, help="SHAKTI Vault Server URL")
    parser.add_argument("--check", action="store_true", help="Check Vault connection and key status")
    parser.add_argument("--status", action="store_true", help="Print Vault Memory status")
    parser.add_argument("--query", type=str, default=None, help="Query Vault for a question")
    args = parser.parse_args()

    client = ShaktiClient(api_key=args.key, vault_url=args.url)
    if args.check:
        v = client.verify()
        print(f"Key Verified: {v.get('api_key')} | Tenant: {v.get('key_name')} | Status: {v.get('status')}")
    if args.status:
        st = client.status()
        print(json.dumps(st, indent=2))
    if args.query:
        res = client.route_query(args.query)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
