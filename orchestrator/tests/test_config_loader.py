import os
import tempfile

from core.config_loader import load_config
from core.models import CameraType

SAMPLE_YAML = """
session_name: "Session_Test"
output_folder: "./out"
cameras:
  - name: "ocr_cam"
    source: "a.mp4"
    camera_type: "ocr"
    target_fps: 5.0
    resize_width: 960
    resize_height: 540
  - name: "cam1"
    source: "b.mp4"
    camera_type: "component"
gap_confidence_threshold: 0.55
resize_width: 848
resize_height: 100
"""


def test_load_config_parses_cameras_and_overrides_defaults():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_YAML)
        path = f.name
    try:
        config = load_config(path)
        assert config.session_name == "Session_Test"
        assert len(config.cameras) == 2
        assert config.cameras[0].camera_type == CameraType.OCR
        assert config.cameras[1].target_fps == 5.0   # default applied
        assert config.gap_confidence_threshold == 0.55
        assert config.ring_buffer_size == 300         # dataclass default untouched

        assert config.cameras[0].resize_width == 960   # per-camera override present
        assert config.cameras[0].resize_height == 540
        assert config.cameras[1].resize_width is None  # falls back to global at use-site
        assert config.cameras[1].resize_height is None
    finally:
        os.unlink(path)
