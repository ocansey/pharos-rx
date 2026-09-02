"""Provider-agnostic chat model construction.

Four back-ends behind ``langchain_core.language_models.BaseChatModel``:
Anthropic, OpenAI, Ollama, and a deterministic ``mock``.

The mock is not a stub. It is a rule-driven generator that produces
well-formed, correctly-cited output from the same context the real models see,
which means the whole graph — triage, retrieval, statistics, synthesis,
verification, repair — is exercised end to end in CI with no key, no network,
and no cost, and the structural assertions in the test suite are real
assertions rather than mocks of assertions. What it cannot do is write well;
that is what the real providers are for.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pharos.config import LLMConfig

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1:8b",
}

_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class DeterministicMockChatModel(BaseChatModel):
    """An offline chat model that produces structurally valid responses.

    It reads the task marker the prompt templates place in the system message
    and responds in the shape that stage expects: a triage verdict, a cited
    synthesis, or a per-claim verification verdict. Output is a pure function of
    the input, so a test that passes once passes forever.
    """

    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "pharos-mock"

    # ------------------------------------------------------------------ #
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "\n".join(str(m.content) for m in messages)
        response = self._respond(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _respond(prompt: str) -> str:
        if "TASK: TRIAGE" in prompt:
            return DeterministicMockChatModel._triage(prompt)
        if "TASK: VERIFY" in prompt:
            return DeterministicMockChatModel._verify(prompt)
        if "TASK: SYNTHESIZE" in prompt:
            return DeterministicMockChatModel._synthesize(prompt)
        return "MOCK: no task marker recognised."

    @staticmethod
    def _triage(prompt: str) -> str:
        personal = re.search(
            r"\b(should i|can i take|my doctor|is it safe for me|how much should i|"
            r"i was prescribed|do i need|diagnose)\b",
            prompt,
            re.I,
        )
        verdict = "PERSONAL_MEDICAL_ADVICE" if personal else "INFORMATIONAL"
        return f"CLASSIFICATION: {verdict}\nRATIONALE: deterministic rule match."

    @staticmethod
    def _synthesize(prompt: str) -> str:
        """Compose an answer from whatever identifiers appear in the context.

        Quoting real fragments of the retrieved units matters: the verification
        node must have something genuine to entail against, or the repair loop
        would be testing itself against its own placeholder text.
        """
        unit_ids = re.findall(r"\[(EU-[\w-]+)\]", prompt)
        stat_ids = re.findall(r"\[(STAT-\d+)\]", prompt)
        seen: list[str] = []
        for uid in unit_ids:
            if uid not in seen:
                seen.append(uid)

        lines = ["What reviewers report:"]
        if stat_ids:
            lines.append(
                f"Across the cohort, the computed summary shows the outcome "
                f"distribution given above [{stat_ids[0]}]."
            )
        for uid in seen[:4]:
            snippet = DeterministicMockChatModel._snippet_for(prompt, uid)
            lines.append(f'One reviewer writes "{snippet}" [{uid}].')
        if not seen and not stat_ids:
            lines.append("INSUFFICIENT_EVIDENCE")
        return "\n".join(lines)

    @staticmethod
    def _snippet_for(prompt: str, unit_id: str) -> str:
        match = re.search(rf"\[{re.escape(unit_id)}\][^\n]*\n\s*\"([^\"]{{0,160}})", prompt)
        if not match:
            return "their experience"
        words = match.group(1).split()
        return " ".join(words[:12]).rstrip(".,;:")

    @staticmethod
    def _verify(prompt: str) -> str:
        claims = re.findall(r"^CLAIM (\d+): (.*)$", prompt, re.M)
        out = []
        for number, claim in claims:
            # A claim is supported iff it carries a citation whose identifier is
            # present in the evidence block above it.
            cited = re.findall(r"\[((?:EU|STAT)-[\w-]+)\]", claim)
            present = [c for c in cited if f"[{c}]" in prompt.split("CLAIM 1:")[0]]
            verdict = "SUPPORTED" if present else "UNSUPPORTED"
            out.append(f"CLAIM {number}: {verdict}")
        return "\n".join(out) if out else "CLAIM 1: SUPPORTED"


# --------------------------------------------------------------------------- #
def build_chat_model(cfg: LLMConfig) -> BaseChatModel:
    """Instantiate the configured chat model.

    Falls back to the mock — loudly, via a warning, never silently — when a
    provider is selected but its key is absent. Silently degrading to a
    different model is how an evaluation ends up reporting numbers from a system
    nobody meant to run.
    """
    provider = cfg.provider
    model_name = cfg.model or DEFAULT_MODELS.get(provider)

    if provider == "mock":
        return DeterministicMockChatModel(temperature=cfg.temperature)

    env_key = _ENV_KEYS.get(provider)
    if env_key and not os.environ.get(env_key):
        import warnings

        warnings.warn(
            f"llm.provider='{provider}' but {env_key} is not set; falling back to the "
            f"deterministic mock. Set {env_key} to generate with {model_name}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DeterministicMockChatModel(temperature=cfg.temperature)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "provider='anthropic' requires: pip install 'pharos-rx[anthropic]'"
            ) from exc
        return ChatAnthropic(
            model=model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "provider='openai' requires: pip install 'pharos-rx[openai]'"
            ) from exc
        return ChatOpenAI(
            model=model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "provider='ollama' requires: pip install 'pharos-rx[ollama]'"
            ) from exc
        return ChatOllama(model=model_name, temperature=cfg.temperature, num_predict=cfg.max_tokens)

    raise ValueError(f"unknown provider: {provider}")


def provider_is_live(cfg: LLMConfig) -> bool:
    """Whether a real generative back-end will actually be used."""
    if cfg.provider == "mock":
        return False
    if cfg.provider == "ollama":
        return True
    return bool(os.environ.get(_ENV_KEYS.get(cfg.provider, ""), ""))
