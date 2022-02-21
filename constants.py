constants = {
    # --- CAMERA CALIBRATION CONSTANTS
    "CALIBRATION_CAMERA_STATIC_PATH": "assets/cam1 - static/calibration.mov",
    "CALIBRATION_CAMERA_MOVING_PATH": "assets/cam2 - moving light/calibration.mp4",
    "CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH": "data/static_intrinsics.xml",
    "CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH": "data/moving_intrinsics.xml",
    "CHESSBOARD_SIZE": (6, 9),
    "CALIBRATION_FRAME_SKIP_INTERVAL": 40,  # We just need some, not all
    # --- ANALYSIS CONSTANTS
    "SQAURE_GRID_DIMENSION": 200,  # It will be a 400x400 square grid inside the marker
    "ALIGNED_VIDEO_FPS": 30,
    "ANALYSIS_FRAME_SKIP": 5,  # It will skip this frames each iteration during analysis
    # ---  DEBUG CONSTANTS
    "STATIC_CAMERA_FEED_WINDOW_TITLE": "Static camera feed",
    "MOVING_CAMERA_FEED_WINDOW_TITLE": "Moving camera feed",
    "LIGHT_DIRECTION_WINDOW_TITLE": "Light direction",
    "LIGHT_DIRECTION_WINDOW_SIZE": 200,
    # ---  INTERACTIVE RELIGHTING CONSTANTS
    "INTERPOLATED_WINDOW_TITLE": "Interpolated Data",
    "INPUT_LIGHT_DIRECTION_WINDOW_TITLE": "Light direction input",
    # ---  FILE NAMES CONSTANTS
    "COIN_1_VIDEO_CAMERA_STATIC_PATH": "assets/cam1 - static/coin1.mov",
    "COIN_1_VIDEO_CAMERA_MOVING_PATH": "assets/cam2 - moving light/coin1.mp4",
    "COIN_2_VIDEO_CAMERA_STATIC_PATH": "assets/cam1 - static/coin2.mov",
    "COIN_2_VIDEO_CAMERA_MOVING_PATH": "assets/cam2 - moving light/coin2.mp4",
    "COIN_3_VIDEO_CAMERA_STATIC_PATH": "assets/cam1 - static/coin3.mov",
    "COIN_3_VIDEO_CAMERA_MOVING_PATH": "assets/cam2 - moving light/coin3.mp4",
    "COIN_4_VIDEO_CAMERA_STATIC_PATH": "assets/cam1 - static/coin4.mov",
    "COIN_4_VIDEO_CAMERA_MOVING_PATH": "assets/cam2 - moving light/coin4.mp4",
    "FILE_1_MOVING_CAMERA_DELAY": 2.724,  # [seconds] (static) 3.609 - 0.885 (moving)
    "FILE_2_MOVING_CAMERA_DELAY": 2.024,  # [seconds] (static) 2.995 - 0.971 (moving)
    "FILE_3_MOVING_CAMERA_DELAY": 2.275,  # [seconds] (static) 3.355 - 1.08 (moving)
    "FILE_4_MOVING_CAMERA_DELAY": 2.015,  # [seconds] (static) 2.960 - 0.945 (moving)
    "COIN_1_ALIGNED_VIDEO_STATIC_PATH": "data/1_static_aligned_video.mov",
    "COIN_1_ALIGNED_VIDEO_MOVING_PATH": "data/1_moving_aligned_video.mp4",
    "COIN_2_ALIGNED_VIDEO_STATIC_PATH": "data/2_static_aligned_video.mov",
    "COIN_2_ALIGNED_VIDEO_MOVING_PATH": "data/2_moving_aligned_video.mp4",
    "COIN_3_ALIGNED_VIDEO_STATIC_PATH": "data/3_static_aligned_video.mov",
    "COIN_3_ALIGNED_VIDEO_MOVING_PATH": "data/3_moving_aligned_video.mp4",
    "COIN_4_ALIGNED_VIDEO_STATIC_PATH": "data/4_static_aligned_video.mov",
    "COIN_4_ALIGNED_VIDEO_MOVING_PATH": "data/4_moving_aligned_video.mp4",
    "COIN_1_EXTRACTED_DATA_FILE_PATH": "data/1_extracted_data.npz",
    "COIN_2_EXTRACTED_DATA_FILE_PATH": "data/2_extracted_data.npz",
    "COIN_3_EXTRACTED_DATA_FILE_PATH": "data/3_extracted_data.npz",
    "COIN_4_EXTRACTED_DATA_FILE_PATH": "data/4_extracted_data.npz",
    "COIN_1_INTERPOLATED_DATA_RBF_FILE_PATH": "data/1_rbf_interpolated_data.npz",
    "COIN_2_INTERPOLATED_DATA_RBF_FILE_PATH": "data/2_rbf_interpolated_data.npz",
    "COIN_3_INTERPOLATED_DATA_RBF_FILE_PATH": "data/3_rbf_interpolated_data.npz",
    "COIN_4_INTERPOLATED_DATA_RBF_FILE_PATH": "data/4_rbf_interpolated_data.npz",
    "COIN_1_INTERPOLATED_DATA_PTM_FILE_PATH": "data/1_ptm_interpolated_data.npz",
    "COIN_2_INTERPOLATED_DATA_PTM_FILE_PATH": "data/2_ptm_interpolated_data.npz",
    "COIN_3_INTERPOLATED_DATA_PTM_FILE_PATH": "data/3_ptm_interpolated_data.npz",
    "COIN_4_INTERPOLATED_DATA_PTM_FILE_PATH": "data/4_ptm_interpolated_data.npz",
}
