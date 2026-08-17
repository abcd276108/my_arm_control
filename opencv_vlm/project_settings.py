"""Central default paths for the PCB proof-of-concept pipeline.

Put camera originals in INPUT_DIR.  Every generated artefact belongs in
OUTPUT_DIR, including the rectified board image used by later stages.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
CONFIG_DIR = PROJECT_DIR / "config"

# Change this one path after downloading Qwen2.5-VL locally.
DEFAULT_MODEL_PATH = Path(r"D:\models\Qwen2.5-VL-7B-Instruct")

# Default filenames; command-line arguments can always override them.
DEFAULT_EMPTY_BOARD_IMAGE = INPUT_DIR / "empty_board.jpg"
DEFAULT_INPUT_MEDIA = INPUT_DIR / "input.jpg"
DEFAULT_REFERENCE_IMAGE = OUTPUT_DIR / "board_rectified.jpg"
DEFAULT_GRID_MAP = OUTPUT_DIR / "grid_map.json"
DEFAULT_CAMERA_PROFILE = CONFIG_DIR / "camera_profile.json"
DEFAULT_ROBOT_CALIBRATION = OUTPUT_DIR / "robot_camera_calibration.json"
