"""The wrapper is the single place cost is counted, so its contract is tested."""
import pytest

from src import llm


@pytest.fixture(autouse=True)
def fake_backend():
    llm.set_provider("fake")
    llm.reset_fake_log()
    yield
    llm.reset_fake_log()


def test_call_returns_text_tokens_usd():
    text, tokens, usd = llm.call("fake-model", "system", "user")
    assert isinstance(text, str) and text
    assert isinstance(tokens, int) and tokens > 0
    assert isinstance(usd, float)


def test_local_models_are_free_not_guessed():
    """An unknown model must cost 0.0, never an invented price."""
    assert llm.price_of("qwen2.5-coder:7b") == (0.0, 0.0)
    assert llm.usd_for("qwen2.5-coder:7b", 1_000_000, 1_000_000) == 0.0


def test_hosted_price_table_arithmetic():
    # 1M input + 1M output on Sonnet 5 = $2 + $10.
    assert llm.usd_for("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert llm.usd_for("claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0)


def test_unknown_provider_is_rejected():
    llm.set_provider("nope")
    with pytest.raises(llm.LLMError, match="unknown provider"):
        llm.call("m", "s", "u")


def test_configure_reads_provider_from_config():
    llm.configure({"provider": "fake"})
    assert llm.current_provider() == "fake"


def test_fake_backend_is_deterministic():
    a = llm.call("m", "rank the entities", "u")[0]
    b = llm.call("m", "rank the entities", "u")[0]
    assert a == b
