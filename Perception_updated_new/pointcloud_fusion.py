import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning(
        "PointCloudFusion: open3d not installed.\n"
        "On Jetson ARM64 install with: pip install open3d-cpu\n"
        "Or build from source — see Dockerfile.perception"
    )


class PointCloudFusion:
    """
    Creates and optionally visualises Open3D point clouds from RGBD frames.

    Fixes vs original:
    - Hardcoded intrinsics replaced — reads from camera.get_intrinsics()
    - OPEN3D_AVAILABLE guard — graceful fallback if open3d not installed
    - create_pointcloud() accepts intrinsics dict so pipeline can pass real values
    - Visualisation gated by SHOW_DISPLAY env var — safe in headless Docker
    - Voxel downsampling added — reduces point cloud size for faster SLAM
    """

    def __init__(self, enable_visualization: bool = False):
        self._available = OPEN3D_AVAILABLE
        self.enable_visualization = (
            enable_visualization
            and OPEN3D_AVAILABLE
            and os.environ.get("SHOW_DISPLAY", "0") == "1"
        )
        self.initialized = False
        self.vis  = None
        self.pcd  = None

        if self.enable_visualization:
            try:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window("3D Map")
                self.pcd = o3d.geometry.PointCloud()
                logger.info("PointCloudFusion: Open3D visualizer opened")
            except Exception as e:
                logger.warning("PointCloudFusion: visualizer failed (%s) — disabled", e)
                self.enable_visualization = False

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def create_pointcloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Optional[dict] = None,
        voxel_size: float = 0.01,
    ):
        """
        Create an Open3D point cloud from aligned RGB + depth frames.

        Args:
            rgb:        BGR uint8 numpy array (H×W×3)
            depth:      uint16 raw depth array (H×W) — raw RealSense values
            intrinsics: {fx, fy, cx, cy, width, height} — from camera.get_intrinsics()
                        If None, uses approximate defaults (NOT recommended)
            voxel_size: downsampling voxel size in metres (0 = no downsampling)

        Returns:
            open3d.geometry.PointCloud, or None if open3d unavailable
        """
        if not self._available:
            logger.debug("PointCloudFusion: open3d not available")
            return None

        try:
            # Convert BGR → RGB for Open3D
            rgb_rgb = rgb[:, :, ::-1].copy()

            o3d_rgb   = o3d.geometry.Image(rgb_rgb.astype(np.uint8))
            o3d_depth = o3d.geometry.Image(depth.astype(np.uint16))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb,
                o3d_depth,
                depth_scale=1000.0,          # RealSense raw unit → metres
                depth_trunc=4.0,             # ignore points beyond 4m
                convert_rgb_to_intensity=False,
            )

            # Use real intrinsics if provided, else defaults
            if intrinsics:
                o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                    int(intrinsics.get("width",  640)),
                    int(intrinsics.get("height", 480)),
                    float(intrinsics["fx"]),
                    float(intrinsics["fy"]),
                    float(intrinsics["cx"]),
                    float(intrinsics["cy"]),
                )
            else:
                logger.warning(
                    "PointCloudFusion: no intrinsics provided — using approximate defaults. "
                    "Pass camera.get_intrinsics() for accurate point cloud."
                )
                o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                    640, 480, 615.0, 615.0, 320.0, 240.0
                )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)

            # Downsample to reduce size for SLAM / world model
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)

            return pcd

        except Exception as e:
            logger.error("PointCloudFusion.create_pointcloud: %s", e)
            return None

    def visualize(self, pcd) -> None:
        """Update the Open3D visualizer with a new point cloud."""
        if not self.enable_visualization or pcd is None:
            return

        try:
            if not self.initialized:
                self.pcd = pcd
                self.vis.add_geometry(self.pcd)
                self.initialized = True
            else:
                self.pcd.points = pcd.points
                self.pcd.colors = pcd.colors
                self.vis.update_geometry(self.pcd)

            self.vis.poll_events()
            self.vis.update_renderer()

        except Exception as e:
            logger.error("PointCloudFusion.visualize: %s", e)
