"""
Universal LLM provider manager for HOT-Step 9000.

Ported from Lireek's ``llm_service.py`` — supports Gemini, OpenAI, Anthropic,
Ollama, LM Studio, and Unsloth Studio.

Configuration is read in this order (first found wins):
1. Lireek DB ``settings`` table  (runtime-editable via the UI)
2. Environment variables / ``.env`` file
3. Hardcoded defaults

Public API
----------
- ``get_provider(name)``  → LLMProvider instance
- ``list_providers()``    → list[ProviderInfo]
- ``init_providers()``    → call once at startup to wire DB settings
"""

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ProviderInfo:
    """Serialisable snapshot of a provider's state."""
    id: str
    name: str
    available: bool
    models: list[str]
    default_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "available": self.available,
            "models": self.models,
            "default_model": self.default_model,
        }


# ── Settings bridge ──────────────────────────────────────────────────────────
# Lazy import from lireek_db to avoid circular imports at module level.

def _get_setting(key: str, default: str = "") -> str:
    """Read a setting from lireek_db, falling back to env then default."""
    try:
        from acestep.api.lireek.lireek_db import get_setting
        val = get_setting(key)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key.upper(), default)


# ── Skip-Thinking state ─────────────────────────────────────────────────────

_skip_thinking_event = threading.Event()


def skip_thinking() -> None:
    """Signal the current LLM streaming call to skip its thinking phase."""
    logger.info("Skip-thinking signal received")
    _skip_thinking_event.set()


def _strip_thinking_blocks(text: str) -> str:
    """Strip Chain-of-Thought thinking blocks from model output."""
    # Standard XML tags (Qwen 3, DeepSeek, etc.)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<analysis>.*?</analysis>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<reflection>.*?</reflection>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL).strip()
    # Unclosed thinking tags
    text = re.sub(
        r'<(?:think|analysis|reasoning|reflection|thought)>.*$',
        '', text, flags=re.DOTALL,
    ).strip()
    # Plain text CoT (LM Studio GGUF quirks)
    pattern = (
        r'^(?:\s*\*+\s*)?'
        r'(?:Thinking Process|Thought Process|Thinking|Reasoning):\s*.*?'
        r'(?:---|[*]{3,}|={3,})\s*'
    )
    match = re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = text[match.end():].strip()
    return text


# ── Provider base class ──────────────────────────────────────────────────────

class LLMProvider:
    id: str = ""
    name: str = ""
    default_model: str = ""
    available_models: list[str] = []

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def to_info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id,
            name=self.name,
            available=self.is_available(),
            models=self.available_models,
            default_model=self.default_model,
        )


# ── Gemini ───────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    id = "gemini"
    name = "Google Gemini"

    @property
    def default_model(self) -> str:
        return _get_setting("gemini_model", "gemini-2.5-flash")

    @property
    def available_models(self) -> list[str]:
        if not self.is_available():
            return [self.default_model]
        try:
            import google.generativeai as genai
            genai.configure(api_key=_get_setting("gemini_api_key"))
            models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if name.startswith("gemini-"):
                        models.append(name)
            if models:
                models.sort(reverse=True)
                return models
        except Exception as e:
            logger.warning("Failed to list Gemini models: %s", e)
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    def is_available(self) -> bool:
        return bool(_get_setting("gemini_api_key"))

    def call(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
             on_chunk: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        import google.generativeai as genai
        genai.configure(api_key=_get_setting("gemini_api_key"))
        model_name = model or self.default_model
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )
        response = gemini_model.generate_content(user_prompt)
        result = response.text
        if on_chunk:
            on_chunk(result)
        return result


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    id = "openai"
    name = "OpenAI / ChatGPT"
    available_models = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
    ]

    @property
    def default_model(self) -> str:
        return _get_setting("openai_model", "gpt-4o-mini")

    def is_available(self) -> bool:
        return bool(_get_setting("openai_api_key"))

    def call(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
             on_chunk: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=_get_setting("openai_api_key"))
        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        result = response.choices[0].message.content or ""
        if on_chunk:
            on_chunk(result)
        return result


# ── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    id = "anthropic"
    name = "Anthropic / Claude"
    available_models = [
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]

    @property
    def default_model(self) -> str:
        return _get_setting("anthropic_model", "claude-3-5-haiku-20241022")

    def is_available(self) -> bool:
        return bool(_get_setting("anthropic_api_key"))

    def call(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
             on_chunk: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=_get_setting("anthropic_api_key"))
        message = client.messages.create(
            model=model or self.default_model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        result = message.content[0].text
        if on_chunk:
            on_chunk(result)
        return result


# ── Ollama ───────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    id = "ollama"
    name = "Ollama (Local)"

    @property
    def default_model(self) -> str:
        return _get_setting("ollama_model", "llama3")

    def _get_base_url(self) -> str:
        return _get_setting("ollama_base_url", "http://localhost:11434").rstrip("/")

    def is_available(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self._get_base_url()}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def get_local_models(self) -> list[str]:
        try:
            import httpx
            resp = httpx.get(f"{self._get_base_url()}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    @property
    def available_models(self) -> list[str]:
        models = self.get_local_models()
        return models or [self.default_model]

    def call(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
             on_chunk: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        import httpx

        model_name = model or self.default_model
        base_url = self._get_base_url()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options: dict = {"num_predict": 8196}
        if kwargs.get("temperature") is not None:
            options["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            options["top_p"] = kwargs["top_p"]

        if on_chunk:
            payload = {"model": model_name, "messages": messages, "stream": True, "options": options}
            chunks: list[str] = []
            skipped = False
            import json as _json
            with httpx.stream("POST", f"{base_url}/api/chat", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = _json.loads(line)
                        text = data.get("message", {}).get("content", "")
                        if text:
                            chunks.append(text)
                            on_chunk(text)
                    except Exception:
                        pass
                    full_so_far = "".join(chunks)
                    if (
                        _skip_thinking_event.is_set()
                        and "<think>" in full_so_far
                        and "</think>" not in full_so_far
                    ):
                        logger.info("Skip-thinking: aborting Ollama stream after %d tokens", len(chunks))
                        skipped = True
                        break
            if skipped:
                return self._continue_after_skip(
                    system_prompt, user_prompt, model_name, "".join(chunks), on_chunk,
                )
            return "".join(chunks)
        else:
            payload = {"model": model_name, "messages": messages, "stream": False, "options": options}
            resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    def _continue_after_skip(
        self, system_prompt: str, user_prompt: str,
        model_name: str, thinking_so_far: str,
        on_chunk: Optional[Callable[[str], None]],
    ) -> str:
        import httpx
        _skip_thinking_event.clear()
        prefix = thinking_so_far.rstrip() + "\n</think>\n\n"
        if on_chunk:
            on_chunk("\n</think>\n\n[Thinking skipped — producing output...]\n\n")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefix},
        ]
        base_url = self._get_base_url()
        if on_chunk:
            payload = {"model": model_name, "messages": messages, "stream": True, "options": {"num_predict": 8196}}
            continuation: list[str] = []
            import json as _json
            with httpx.stream("POST", f"{base_url}/api/chat", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = _json.loads(line)
                        text = data.get("message", {}).get("content", "")
                        if text:
                            continuation.append(text)
                            on_chunk(text)
                    except Exception:
                        pass
            return prefix + "".join(continuation)
        else:
            payload = {"model": model_name, "messages": messages, "stream": False, "options": {"num_predict": 8196}}
            resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            return prefix + resp.json()["message"]["content"]

    def to_info(self) -> ProviderInfo:
        local_models = self.get_local_models()
        return ProviderInfo(
            id=self.id, name=self.name,
            available=self.is_available(),
            models=local_models or [self.default_model],
            default_model=local_models[0] if local_models else self.default_model,
        )


# ── LM Studio ───────────────────────────────────────────────────────────────

class LMStudioProvider(LLMProvider):
    id = "lmstudio"
    name = "LM Studio"

    @property
    def default_model(self) -> str:
        return _get_setting("lmstudio_model", "")

    def _get_base_url(self) -> str:
        return _get_setting("lmstudio_base_url", "http://localhost:1234/v1").rstrip("/")

    def _client(self):
        from openai import OpenAI
        return OpenAI(base_url=self._get_base_url(), api_key="lm-studio")

    def is_available(self) -> bool:
        try:
            import httpx
            base_url = self._get_base_url()
            # LM Studio serves /v1/models — use base_url as-is
            resp = httpx.get(f"{base_url}/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        try:
            client = self._client()
            models = client.models.list()
            return sorted([m.id for m in models.data], reverse=True)
        except Exception as e:
            logger.warning("Failed to list LM Studio models: %s", e)
            return []

    @property
    def available_models(self) -> list[str]:
        models = self.get_available_models()
        return models or ([self.default_model] if self.default_model else [])

    def call(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
             on_chunk: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        client = self._client()
        model_name = model or self.default_model
        if not model_name:
            available = self.get_available_models()
            if available:
                model_name = available[0]
            else:
                raise ValueError("No models loaded in LM Studio")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        sampling: dict = {}
        if kwargs.get("temperature") is not None:
            sampling["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            sampling["top_p"] = kwargs["top_p"]

        if on_chunk:
            stream = client.chat.completions.create(
                model=model_name, messages=messages, stream=True, **sampling,
            )
            chunks: list[str] = []
            skipped = False
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    chunks.append(text)
                    on_chunk(text)
                full_so_far = "".join(chunks)
                if (
                    _skip_thinking_event.is_set()
                    and "<think>" in full_so_far
                    and "</think>" not in full_so_far
                ):
                    logger.info("Skip-thinking: aborting LM Studio stream after %d tokens", len(chunks))
                    skipped = True
                    try:
                        stream.close()
                    except Exception:
                        pass
                    break
            if skipped:
                return self._continue_after_skip(
                    system_prompt, user_prompt, model_name, "".join(chunks), on_chunk,
                )
            return "".join(chunks)
        else:
            response = client.chat.completions.create(
                model=model_name, messages=messages, **sampling,
            )
            return response.choices[0].message.content or ""

    def _continue_after_skip(
        self, system_prompt: str, user_prompt: str,
        model_name: str, thinking_so_far: str,
        on_chunk: Optional[Callable[[str], None]],
    ) -> str:
        _skip_thinking_event.clear()
        prefix = thinking_so_far.rstrip() + "\n</think>\n\n"
        if on_chunk:
            on_chunk("\n</think>\n\n[Thinking skipped — producing output...]\n\n")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefix},
        ]
        client = self._client()
        if on_chunk:
            stream = client.chat.completions.create(
                model=model_name, messages=messages, stream=True,
            )
            continuation: list[str] = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    continuation.append(text)
                    on_chunk(text)
            return prefix + "".join(continuation)
        else:
            response = client.chat.completions.create(
                model=model_name, messages=messages,
            )
            return prefix + (response.choices[0].message.content or "")

    def to_info(self) -> ProviderInfo:
        models = self.get_available_models()
        return ProviderInfo(
            id=self.id, name=self.name,
            available=self.is_available(),
            models=models or ([self.default_model] if self.default_model else []),
            default_model=models[0] if models else self.default_model,
        )


# ── Unsloth Studio ───────────────────────────────────────────────────────────

class UnslothProvider(LLMProvider):
    id = "unsloth"
    name = "Unsloth Studio"

    _cached_token: str = ""
    _token_expiry: float = 0.0

    @property
    def default_model(self) -> str:
        return _get_setting("unsloth_model", "")

    def _get_base_url(self) -> str:
        return _get_setting("unsloth_base_url", "http://127.0.0.1:8888").rstrip("/")

    def _get_credentials(self) -> tuple[str, str]:
        return (
            _get_setting("unsloth_username", ""),
            _get_setting("unsloth_password", ""),
        )

    def _authenticate(self) -> str:
        import time
        if self._cached_token and time.time() < self._token_expiry - 60:
            return self._cached_token

        username, password = self._get_credentials()
        if not username or not password:
            raise ValueError("Unsloth Studio username/password not configured")

        import httpx
        base_url = self._get_base_url()
        for payload in [
            {"email": username, "password": password},
            {"username": username, "password": password},
        ]:
            try:
                resp = httpx.post(f"{base_url}/api/auth/login", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    token = data.get("access_token") or data.get("token")
                    if token:
                        import base64
                        import json as _json
                        try:
                            payload_b64 = token.split(".")[1]
                            payload_b64 += "=" * (4 - len(payload_b64) % 4)
                            jwt_data = _json.loads(base64.urlsafe_b64decode(payload_b64))
                            self._token_expiry = float(jwt_data.get("exp", time.time() + 3600))
                        except Exception:
                            self._token_expiry = time.time() + 3600
                        self._cached_token = token
                        logger.info("Authenticated with Unsloth Studio (expires in %ds)",
                                    int(self._token_expiry - time.time()))
                        return token
            except Exception as e:
                logger.debug("Unsloth auth attempt failed: %s", e)

        raise ValueError("Failed to authenticate with Unsloth Studio — check username/password")

    def _client(self):
        from openai import OpenAI
        token = self._authenticate()
        return OpenAI(
            base_url=f"{self._get_base_url()}/v1",
            api_key=token,
            timeout=180.0,
        )

    def is_available(self) -> bool:
        username, password = self._get_credentials()
        if not username or not password:
            return False
        try:
            token = self._authenticate()
            import httpx
            resp = httpx.get(
                f"{self._get_base_url()}/v1/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=3,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        try:
            client = self._client()
            models = client.models.list()
            return sorted([m.id for m in models.data], reverse=True)
        except Exception as e:
            logger.warning("Failed to list Unsloth Studio models: %s", e)
            return []

    @property
    def available_models(self) -> list[str]:
        models = self.get_available_models()
        return models or ([self.default_model] if self.default_model else [])

    def call(self, system_prompt: str, user_prompt: str, model: Optional[str] = None,
             on_chunk: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        client = self._client()
        model_name = model or self.default_model
        if not model_name:
            available = self.get_available_models()
            if available:
                model_name = available[0]
            else:
                raise ValueError("No models loaded in Unsloth Studio")

        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        chunks: list[str] = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                chunks.append(text)
                if on_chunk:
                    on_chunk(text)
        return "".join(chunks)

    def to_info(self) -> ProviderInfo:
        models = self.get_available_models()
        return ProviderInfo(
            id=self.id, name=self.name,
            available=self.is_available(),
            models=models or ([self.default_model] if self.default_model else []),
            default_model=models[0] if models else self.default_model,
        )


# ── Registry ─────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, LLMProvider] = {
    "gemini": GeminiProvider(),
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "ollama": OllamaProvider(),
    "lmstudio": LMStudioProvider(),
    "unsloth": UnslothProvider(),
}


def get_provider(name: str) -> LLMProvider:
    """Get a provider by its ID. Raises ValueError if unknown."""
    provider = _PROVIDERS.get(name)
    if provider is None:
        raise ValueError(f"Unknown LLM provider: '{name}'. Choose from: {list(_PROVIDERS.keys())}")
    return provider


def list_providers() -> list[ProviderInfo]:
    """Return info snapshots for all registered providers."""
    return [p.to_info() for p in _PROVIDERS.values()]


def get_default_provider_name() -> str:
    """Return the configured default provider name."""
    return _get_setting("default_llm_provider", "gemini")


def init_providers() -> None:
    """Initialise the provider system. Call once at app startup."""
    logger.info(
        "LLM provider manager initialised — %d providers registered: %s",
        len(_PROVIDERS), ", ".join(_PROVIDERS.keys()),
    )
