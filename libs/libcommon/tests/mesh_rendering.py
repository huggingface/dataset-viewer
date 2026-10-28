# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Authors.

"""Detect if the platform can render 3D thumbnails, and skip the tests which need it if it can't.

The 3D thumbnails are rendered offscreen by pyrender, which needs an OpenGL context: EGL (with the
mesa software rasterizer when there is no GPU) or OSMesa. Neither is installed by default, hence we
cannot require it for the whole test suite: see the "mesh" entry of the datasets fixtures, and the
mesh tests which call ensure_mesh_rendering().
"""

import os
from functools import lru_cache

from pytest import skip

# EGL is the only platform that renders the alpha channel, so the background is transparent with it
# (see libcommon.viewer_utils.asset._render_trimesh_scene_to_png). It's only set if the environment
# didn't choose a platform yet, as PyOpenGL reads it when it's imported.
PREFERRED_PLATFORM = "egl"

SKIP_REASON = (
    "The platform cannot render 3D thumbnails with a transparent background: pyrender needs an "
    "offscreen OpenGL context which renders the alpha channel. Install EGL (e.g the libegl1 and "
    "libegl-mesa0 packages) to run these tests."
)


@lru_cache(maxsize=1)
def mesh_rendering_available() -> bool:
    """Render an empty scene offscreen, and return False if the platform can't render it in RGBA."""
    os.environ.setdefault("PYOPENGL_PLATFORM", PREFERRED_PLATFORM)
    try:
        import numpy as np
        import pyrender

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])  # a fully transparent background
        scene.add(pyrender.PerspectiveCamera(yfov=1.0), pose=np.eye(4))  # pyrender requires a camera
        renderer = pyrender.OffscreenRenderer(viewport_width=8, viewport_height=8)
        try:
            color_image, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        # the alpha channel must be rendered: with an opaque background the thumbnails are not transparent
        return color_image.shape[2] == 4 and bool(color_image[0, 0, 3] == 0)
    except Exception:
        return False


def ensure_mesh_rendering() -> None:
    """Skip the running test if the platform cannot render 3D thumbnails."""
    if not mesh_rendering_available():
        skip(SKIP_REASON)
