import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DepthProcessor:
    """
    Extracts reliable depth estimates from a raw RealSense depth frame.

    Depth frame values are uint16 raw units.
    Multiply by camera.depth_scale to get metres.

    Fixes vs original:
    - get_bbox_depth() no longer divides by 1000.0 — returns raw median value
      so the caller (perception_pipeline) multiplies by depth_scale once.
      Original divided by 1000 AND pipeline multiplied by depth_scale → double scaling.
    - Percentile filter: ignores top/bottom 10% of depth values in ROI
      to handle occlusion and sensor noise better than plain median.
    - Returns (depth_raw, valid_ratio) so pipeline can skip low-confidence depths.
    """

    # Reject ROIs where fewer than this fraction of pixels have valid depth
    MIN_VALID_RATIO = 0.1

    def get_bbox_depth(
        self,
        depth_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        """
        Get robust depth estimate for a bounding box region.

        Args:
            depth_frame: raw uint16 depth array from RealSense (NOT divided by 1000)
            bbox:        (x1, y1, x2, y2) pixel coordinates

        Returns:
            Raw depth value in sensor units (uint16).
            Caller must multiply by camera.depth_scale to get metres.
            Returns None if depth is unavailable or unreliable.

        NOTE: Original divided by 1000.0 here. This caused double-scaling in
        perception_pipeline.py which then also multiplied by depth_scale (~0.001).
        Result was depth ~1000x too small → completely wrong 3D positions.
        """
        x1, y1, x2, y2 = bbox
        roi = depth_frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Only use valid (non-zero) depth pixels
        valid = roi[roi > 0]
        valid_ratio = len(valid) / roi.size

        if valid_ratio < self.MIN_VALID_RATIO:
            logger.debug(
                "DepthProcessor: low valid ratio %.2f at bbox %s — skipping",
                valid_ratio, bbox
            )
            return None

        # Use 40th percentile — more robust than median for partially occluded objects
        # (median can land on background; 40th percentile biases toward closer surface)
        depth_raw = float(np.percentile(valid, 40))

        if depth_raw <= 0:
            return None

        return depth_raw

    def get_depth_at_point(
        self,
        depth_frame: np.ndarray,
        u: int,
        v: int,
        kernel: int = 5,
    ) -> Optional[float]:
        """
        Get depth at a single pixel with a small neighbourhood average.
        Useful for getting depth at the centroid of a tracked object.

        Returns raw uint16 value, or None if invalid.
        """
        h, w = depth_frame.shape
        half = kernel // 2

        y1 = max(0, v - half)
        y2 = min(h, v + half + 1)
        x1 = max(0, u - half)
        x2 = min(w, u + half + 1)

        patch = depth_frame[y1:y2, x1:x2]
        valid = patch[patch > 0]

        if len(valid) == 0:
            return None

        return float(np.median(valid))
