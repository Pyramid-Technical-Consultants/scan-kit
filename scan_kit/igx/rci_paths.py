"""RCI controller IO paths for plan runner."""

from __future__ import annotations

# Control point upload / validation
POINTS_UPLOAD = "rci/control_point_table/points_upload"
POINTS_UPLOAD_TARGET = "rci/control_point_table/points_upload/target"
POINTS_LOAD_STATE = "rci/control_point_table/points_load_state"
POINTS_VALID = "rci/rci_controller/points_valid"

DEFAULT_CONTROL_POINTS_PATH = "/root/config/control/control_points.csv"

# Run controls
START_BUTTON = "rci/rci_controller/start_button"
PAUSE_BUTTON = "rci/rci_controller/pause_button"
STOP_BUTTON = "rci/rci_controller/stop_button"
RESET_BUTTON = "rci/rci_controller/reset_button"

# Controller state / progress
STATE = "rci/rci_controller/state"
PROGRESS = "rci/rci_controller/progress"
POINT_PROGRESS = "rci/rci_controller/point_progress"
CONTROL_POINT_INDEX = "rci/rci_controller/control_point_index"
CONTROL_POINT_COUNT = "rci/rci_controller/control_point_count"
POINT_ENERGY = "rci/rci_controller/point_energy"
POINT_LAYER_ID = "rci/rci_controller/point_layer_id"
TIME_ELAPSED = "rci/rci_controller/time_elapsed"
TREATMENT_ACTIVE = "rci/rci_controller/treatment_active"
READY_PERMIT = "rci/rci_controller/ready_permit"
READY_PERMIT_REASON = "rci/rci_controller/ready_permit/revoke_reason"

# Map manager permits
COMBINED_START_PERMIT = "rci/map_manager/combined_start_permit"
COMBINED_STOP_PERMIT = "rci/map_manager/combined_stop_permit"
COMBINED_STATE = "rci/map_manager/combined_state"
COMBINED_POINTS_OK = "rci/map_manager/combined_points_ok"

# Session storage
SESSION_DIRECTORY = "admin/storage/session/directory"

# points_load_state: 0 idle, 1 loading, 2 success, other error
LOAD_STATE_SUCCESS = 2

STATUS_IO_PATHS: tuple[str, ...] = (
    STATE,
    PROGRESS,
    POINT_PROGRESS,
    CONTROL_POINT_INDEX,
    CONTROL_POINT_COUNT,
    POINT_ENERGY,
    POINT_LAYER_ID,
    TIME_ELAPSED,
    TREATMENT_ACTIVE,
    READY_PERMIT,
    READY_PERMIT_REASON,
    POINTS_VALID,
    POINTS_LOAD_STATE,
    COMBINED_START_PERMIT,
    COMBINED_STOP_PERMIT,
    COMBINED_STATE,
    COMBINED_POINTS_OK,
)
