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
    logger.warning("langchain_openai not installed — pip install langchain-openai")


class LLMClient:
    """
    Jetson-optimized LLM client:
    - Primary: Ollama (Gemma 3 1B — fits 8GB Jetson Orin Nano)
    - Fallback: OpenAI gpt-4o-mini (optional, set OPENAI_API_KEY)
    """

    DEFAULT_TIMEOUT = 30
    MAX_RETRIES     = 2
    RETRY_DELAY     = 2

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout     = timeout
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
            logger.warning("LLMClient: no LLM available — fallback mode")

    # ------------------------------------------------------------------ #
    #  PUBLIC API (used by planner)
    # ------------------------------------------------------------------ #

    def invoke(self, prompt: str) -> Optional[str]:
        """
        Main function used by HighLevelPlanner
        Returns clean text or None
        """
        if not self._available or self._llm is None:
            logger.warning("LLMClient: not available")
            return None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._llm.invoke(prompt)
                return self._extract_text(response)

            except Exception as e:
                logger.warning("LLM error (attempt %d): %s", attempt, e)
                time.sleep(self.RETRY_DELAY)

        return None

    # ------------------------------------------------------------------ #
    #  INIT LOCAL (OLLAMA)
    # ------------------------------------------------------------------ #

    def _try_local(self):
        if not OPENAI_AVAILABLE:
            return

        url   = os.environ.get("LOCAL_LLM_URL",   "http://localhost:11434/v1")
        model = os.environ.get("LOCAL_LLM_MODEL", "gemma3:1b")

        try:
            # Quick ping
            test_llm = ChatOpenAI(
                model=model,
                base_url=url,
                api_key="ollama",
                temperature=0,
                timeout=30,
                max_retries=0,
            )
            test_llm.invoke("hi")

            # Actual client
            self._llm = ChatOpenAI(
                model=model,
                base_url=url,
                api_key="ollama",
                temperature=0,
                timeout=self.timeout,
                max_retries=0,
            )

            self._available = True
            self._mode = "local"

            logger.info("LLMClient: connected to Ollama (%s)", model)

        except Exception as e:
            logger.info("LLMClient: Ollama not available (%s)", e)

    # ------------------------------------------------------------------ #
    #  INIT CLOUD (OPTIONAL)
    # ------------------------------------------------------------------ #

    def _try_cloud(self):
        if not OPENAI_AVAILABLE:
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return

        try:
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                timeout=self.timeout,
                max_retries=0,
            )
            self._available = True
            self._mode = "cloud"

            logger.info("LLMClient: using cloud fallback")

        except Exception as e:
            logger.error("LLMClient: cloud failed (%s)", e)

    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response) -> str:
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()

    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def mode(self) -> str:
        return self._mode