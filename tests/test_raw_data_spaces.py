"""Offline tests of the raw DROID dataset's spaces (no recordings needed)."""

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pyzed")
pytest.importorskip("serde")

from unienv_droid_data.raw.raw_data import _get_frame_space  # noqa: E402

H, W = 4, 5


@pytest.mark.parametrize("read_right", [False, True])
def test_frame_space_builds(read_right):
    space = _get_frame_space(H, W, read_right=read_right)
    obs = space["steps"]["observation"]
    for cam in ("exterior_image_1", "exterior_image_2", "wrist_image"):
        assert obs[f"{cam}_left"].shape == (H, W, 3)
        assert (f"{cam}_right" in obs.spaces) == read_right
    for cam in ("exterior_depth_1", "exterior_depth_2", "wrist_depth"):
        assert obs[f"{cam}_left"].shape == (H, W)
        assert obs[f"{cam}_left"].dtype == np.float32
    assert ("wrist_camera_right" in obs["camera_intrinsics"].spaces) == read_right
