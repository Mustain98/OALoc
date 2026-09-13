# src/llm.py
#
# TAJ owns this. Everyone calls the model through it, so cost is measured in
# exactly one place (00_PROJECT_GUIDE.md §7: "if cost isn't measured, the project
# has no result").
#
# The public signature is frozen:
#
#     call(model, system, user, max_tokens) -> (text, tokens, usd)
#
# Behind it sits a pluggable backend so the experiment is provider-independent.
# Switching providers is one line in config.yaml; no calling code changes.
import os
import time

_DEFAULT_PROVIDER = os.environ.get("QALOC_LLM_PROVIDER", "ollama")
_CFG: dict = {}

# Rough per-Mtoken USD, (input, output). Local models are free — that is the point
# of running them. The hosted prices are kept so a later hosted run reports real
# dollars without touching this file's callers.
_PRICE = {
    # Anthropic (per 1M tokens, as of 2026-06)
    "claude-opus-5":     (5.0, 25.0),
    "claude-sonnet-5":   (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5":  (1.0,  5.0),
}
_DEFAULT_PRICE = (0.0, 0.0)   # unknown model => assume local/free, never invent a cost


class LLMError(RuntimeError):
    """Raised when the backend cannot be reached or returns something unusable."""


def configure(cfg: dict) -> None:
    """Point the module at a loaded config.yaml. Called once by the run scripts."""
    global _CFG, _DEFAULT_PROVIDER
    _CFG = cfg or {}
    _DEFAULT_PROVIDER = _CFG.get("provider", _DEFAULT_PROVIDER)


def set_provider(provider: str) -> None:
    """Override the backend for this process (used by --provider and by tests)."""
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider


def current_provider() -> str:
    return _DEFAULT_PROVIDER


def price_of(model: str) -> tuple[float, float]:
    return _PRICE.get(model, _DEFAULT_PRICE)


def usd_for(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = price_of(model)
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def call(model: str, system: str, user: str, max_tokens: int = 1024):
    """Returns (text, tokens, usd). The single place where all cost is counted."""
    provider = _DEFAULT_PROVIDER
    if provider == "ollama":
        return _call_ollama(model, system, user, max_tokens)
    if provider == "fake":
        return _call_fake(model, system, user, max_tokens)
    if provider == "anthropic":
        return _call_anthropic(model, system, user, max_tokens)
    raise LLMError(f"unknown provider {provider!r} (expected ollama | fake | anthropic)")


# --------------------------------------------------------------------------- #
# Backend: Ollama (local, free)
# --------------------------------------------------------------------------- #
def _call_ollama(model: str, system: str, user: str, max_tokens: int):
    import requests   # imported lazily so the fake backend needs no dependencies

    oc = _CFG.get("ollama", {}) or {}
    host = oc.get("host", "http://localhost:11434").rstrip("/")
    timeout = oc.get("timeout", 600)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            # num_predict caps the reply. num_ctx must be set explicitly: Ollama's
            # default (~4096) silently truncates the prompt, which would flatten the
            # hops=1 / hops=3 distinction that IS our independent variable.
            "num_predict": max_tokens,
            "num_ctx": oc.get("num_ctx", 16384),
            # Deliberately NOT setting temperature. max_samples is one of the three
            # effort knobs and only works if repeated passes differ; pinning
            # temperature=0 would make every sample identical and turn the knob into
            # a silent no-op.
        },
    }

    last_err = None
    data = None
    for attempt in range(3):
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
            if r.status_code == 404:
                # Not transient: the model isn't pulled. Retrying wastes time and
                # buries the one thing the user needs to be told.
                raise LLMError(f"model {model!r} is not available in Ollama. "
                               f"Pull it first: `ollama pull {model}`")
            r.raise_for_status()
            data = r.json()
            break
        except requests.exceptions.ConnectionError as e:
            raise LLMError(
                f"cannot reach Ollama at {host}. Start it with `ollama serve`."
            ) from e
        except LLMError:
            raise
        except Exception as e:                      # transient HTTP / timeout
            last_err = e
            time.sleep(2 ** attempt)
    if data is None:
        raise LLMError(f"Ollama request failed after 3 attempts: {last_err}")

    text = (data.get("message") or {}).get("content", "") or ""
    tin = int(data.get("prompt_eval_count") or 0)
    tout = int(data.get("eval_count") or 0)
    return text, tin + tout, usd_for(model, tin, tout)


# --------------------------------------------------------------------------- #
# Backend: fake (deterministic, offline — lets the whole test suite run with no
# model installed, no network and no API key)
# --------------------------------------------------------------------------- #
_FAKE_CALLS: list[dict] = []


def fake_call_log() -> list[dict]:
    """Every fake call made so far. Tests assert on this (e.g. max_samples honoured)."""
    return _FAKE_CALLS


def reset_fake_log() -> None:
    _FAKE_CALLS.clear()


def _call_fake(model: str, system: str, user: str, max_tokens: int):
    _FAKE_CALLS.append({"model": model, "system": system, "user": user})

    if "keyword" in system.lower():
        text = "cache, Cache.get, KeyError, store"
    else:
        # Intentionally messy: numbered, bulleted, fenced and prose lines, mirroring
        # what a small local model actually emits. The agent's parser must survive it.
        text = (
            "Here are the most likely locations:\n"
            "```\n"
            "1. src/cache.py:Cache.get\n"
            "- src/cache.py:helper\n"
            "* `src/store.py:lookup`\n"
            "```\n"
            "Hope this helps!"
        )

    # Token counts are approximated so cost arithmetic is still exercised offline.
    tin = max(1, (len(system) + len(user)) // 4)
    tout = max(1, len(text) // 4)
    return text, tin + tout, usd_for(model, tin, tout)


# --------------------------------------------------------------------------- #
# Backend: Anthropic (hosted, billed) — extension point, unused by default.
# --------------------------------------------------------------------------- #
def _call_anthropic(model: str, system: str, user: str, max_tokens: int):
    import anthropic   # lazy: not a hard dependency of the local-model setup

    client = anthropic.Anthropic()   # resolves ANTHROPIC_API_KEY / auth profile
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    tin = r.usage.input_tokens + (getattr(r.usage, "cache_read_input_tokens", 0) or 0)
    tout = r.usage.output_tokens
    return text, tin + tout, usd_for(model, tin, tout)
