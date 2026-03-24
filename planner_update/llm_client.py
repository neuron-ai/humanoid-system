import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("langchain_openai not installed")


class LLMClient:
    """
    LLM client with hybrid local/cloud support.

    Priority:
      1. Local Jetson LLM  — Gemma 3 1B via jetson-containers (free, offline, ~100 tok/s)
      2. Cloud LLM         — GPT-4o-mini via OpenAI (fallback, needs internet)

    Start local LLM on Jetson first:
      jetson-containers run $(autotag local_llm) \
        --env MODEL=google/gemma-3-1b-it \
        --env QUANT=q4 \
        --publish 9000:9000

    Environment variables:
      LLM_MODE         = auto | local | cloud  (default: auto)
      LOCAL_LLM_URL    = http://localhost:9000/v1
      LOCAL_LLM_MODEL  = google/gemma-3-1b-it
      LLM_MODEL        = gpt-4o-mini
      OPENAI_API_KEY   = sk-...
    """

    DEFAULT_TIMEOUT  = 15
    MAX_RETRIES      = 3
    RETRY_DELAY      = 2

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, max_retries: int = MAX_RETRIES):
        self.timeout     = timeout
        self.max_retries = max_retries
        self._llm        = None
        self._available  = False
        self._mode       = "none"

        mode = os.environ.get("LLM_MODE", "auto").lower()

        if mode in ("local", "auto"):
            self._try_local()

        if not self._available and mode in ("cloud", "auto"):
            self._try_cloud()

        if self._available:
            logger.info("LLMClient: ready — mode=%s", self._mode)
        else:
            logger.warning("LLMClient: no LLM available — rule-based fallback only")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_llm(self):
        """Return raw LangChain LLM object."""
        return self._llm

    def invoke(self, prompt: str) -> Optional[str]:
        """
        Call LLM with automatic retry on transient errors.
        Returns response string, or None on failure.
        """
        if not self._available or self._llm is None:
            logger.warning("LLMClient: not available — returning None")
            return None

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("LLMClient [%s]: attempt %d/%d", self._mode, attempt, self.max_retries)
                response = self._llm.invoke(prompt)
                content  = self._extract_content(response)
                logger.debug("LLMClient: success on attempt %d", attempt)
                return content
            except Exception as e:
                last_error = e
                logger.warning("LLMClient: attempt %d failed (%s)", attempt, e)
                if attempt < self.max_retries:
                    time.sleep(self.RETRY_DELAY)

        logger.error("LLMClient: all %d attempts failed. Last: %s", self.max_retries, last_error)
        return None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def mode(self) -> str:
        """Returns 'local', 'cloud', or 'none'."""
        return self._mode

    # ------------------------------------------------------------------ #
    #  Init helpers
    # ------------------------------------------------------------------ #

    def _try_local(self) -> None:
        """Try connecting to local Jetson LLM (jetson-containers local_llm)."""
        if not OPENAI_AVAILABLE:
            return
        url   = os.environ.get("LOCAL_LLM_URL",   "http://localhost:9000/v1")
        model = os.environ.get("LOCAL_LLM_MODEL",  "google/gemma-3-1b-it")
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                openai_api_base=url,
                openai_api_key="none",          # no key needed for local
                request_timeout=5,              # short timeout for ping test
                max_retries=0,
            )
            llm.invoke("hi")                    # ping — raises if server not up
            self._llm       = ChatOpenAI(       # re-init with real timeout
                model=model,
                temperature=0,
                openai_api_base=url,
                openai_api_key="none",
                request_timeout=self.timeout,
                max_retries=0,
            )
            self._available = True
            self._mode      = "local"
            logger.info("LLMClient: local LLM connected at %s model=%s", url, model)
        except Exception as e:
            logger.info("LLMClient: local LLM not available (%s) — trying cloud", e)

    def _try_cloud(self) -> None:
        """Try OpenAI cloud API."""
        if not OPENAI_AVAILABLE:
            return
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("LLMClient: OPENAI_API_KEY not set — cloud unavailable")
            return
        try:
            self._llm = ChatOpenAI(
                model=model,
                temperature=0,
                request_timeout=self.timeout,
                max_retries=0,
            )
            self._available = True
            self._mode      = "cloud"
            logger.info("LLMClient: cloud LLM ready — model=%s", model)
        except Exception as e:
            logger.error("LLMClient: cloud init failed (%s)", e)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_content(response) -> str:
        if hasattr(response, "content"):
            return response.content
        return str(response)
