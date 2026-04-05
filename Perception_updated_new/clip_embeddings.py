import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.error(
        "CLIPEncoder: 'clip' not found. Install with:\n"
        "  pip install git+https://github.com/openai/CLIP.git\n"
        "Do NOT use 'pip install clip' — that is a different unrelated package."
    )


class CLIPEncoder:
    """
    OpenAI CLIP image encoder.
    Produces 512-dim L2-normalised embeddings (ViT-B/32).

    Install:
        pip install git+https://github.com/openai/CLIP.git

    Fixes vs original:
    - Graceful ImportError with clear install message
    - encode_image() returns None (not crash) when CLIP unavailable
    - encode_text() added — used by world_model.semantic_search()
    - try/except around encode — bad crops don't crash the pipeline
    """

    EMBEDDING_DIM = 512   # ViT-B/32 output dimension

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model      = None
        self.preprocess = None
        self.device     = "cpu"
        self._available = False

        if not CLIP_AVAILABLE:
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self._available = True
            logger.info("CLIPEncoder: loaded %s on %s", model_name, self.device)
        except Exception as e:
            logger.error("CLIPEncoder: failed to load model (%s)", e)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def encode_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode a BGR numpy image crop to a 512-dim float32 numpy array.
        Returns None if CLIP unavailable or encoding fails.
        """
        if not self._available or image is None:
            return None

        try:
            rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalise

            return emb.cpu().numpy()[0].astype(np.float32)

        except Exception as e:
            logger.error("CLIPEncoder.encode_image: %s", e)
            return None

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a text string to a 512-dim float32 numpy array.
        Used by world_model.semantic_search() to compare text to image embeddings.
        Returns None if CLIP unavailable.
        """
        if not self._available:
            return None

        try:
            tokens = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            return emb.cpu().numpy()[0].astype(np.float32)

        except Exception as e:
            logger.error("CLIPEncoder.encode_text: %s", e)
            return None

    @property
    def is_available(self) -> bool:
        return self._available
