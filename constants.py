constants = {
    # --- COINS ASSETS FILE NAMES AND DELAY BETWEEN FOOTAGE
    "CALIBRATION_CAMERA_STATIC_PATH": "assets/coins/cam1 - static/calibration.mov",
    "CALIBRATION_CAMERA_MOVING_PATH": "assets/coins/cam2 - moving light/calibration.mp4",
    "COIN_1_VIDEO_CAMERA_STATIC_PATH": "assets/coins/cam1 - static/coin1.mov",
    "COIN_1_VIDEO_CAMERA_MOVING_PATH": "assets/coins/cam2 - moving light/coin1.mp4",
    "COIN_2_VIDEO_CAMERA_STATIC_PATH": "assets/coins/cam1 - static/coin2.mov",
    "COIN_2_VIDEO_CAMERA_MOVING_PATH": "assets/coins/cam2 - moving light/coin2.mp4",
    "COIN_3_VIDEO_CAMERA_STATIC_PATH": "assets/coins/cam1 - static/coin3.mov",
    "COIN_3_VIDEO_CAMERA_MOVING_PATH": "assets/coins/cam2 - moving light/coin3.mp4",
    "COIN_4_VIDEO_CAMERA_STATIC_PATH": "assets/coins/cam1 - static/coin4.mov",
    "COIN_4_VIDEO_CAMERA_MOVING_PATH": "assets/coins/cam2 - moving light/coin4.mp4",
    "FILE_1_MOVING_CAMERA_DELAY": 2.724,  # [seconds] (static) 3.609 - 0.885 (moving)
    "FILE_2_MOVING_CAMERA_DELAY": 2.024,  # [seconds] (static) 2.995 - 0.971 (moving)
    "FILE_3_MOVING_CAMERA_DELAY": 2.275,  # [seconds] (static) 3.355 - 1.08 (moving)
    "FILE_4_MOVING_CAMERA_DELAY": 2.015,  # [seconds] (static) 2.960 - 0.945 (moving)
    # --- CAMERA CALIBRATION CONSTANTS
    "CHESSBOARD_SIZE": (6, 9),
    "CALIBRATION_FRAME_SKIP_INTERVAL": 40,  # We just need some, not all
    # --- ANALYSIS CONSTANTS
    "SQAURE_GRID_DIMENSION": 200,  # It will be a 200x200 square grid inside the marker
    "LIGHT_DIRECTION_WINDOW_SIZE": 100,  # There will be 100x100 possible light directions
    "LIGHT_DIRECTION_WINDOW_SCALE": 2,  # The light window will be 2 times the size
    "ALIGNED_VIDEO_FPS": 30,
    "ANALYSIS_FRAME_SKIP": 5,  # It will skip this frames each iteration during analysis
    "COINS_TEST_N_LIGHTS": 5,
    "SSIM_GAUSSIAN_KERNEL_SIZE": 11,
    # ---  PCA MODEL CONSTANTS
    "PCA_BATCH_SIZE": 64,
    "PCA_LEARNING_RATE": 0.0001,
    "PCA_N_EPOCHS": 40,
    "PCA_ORTHOGONAL_BASES": 8,
    "PCA_H": 10,
    "PCA_SIGMA": 0.3,
    "PCA_MODEL_INPUT_SIZE": 8 + (2 * 10),  # PCA_ORTHOGONAL_BASES + (2 * H)
    "GAUSSIAN_MATRIX_FILE_PATH": "data/gaussian_matrix.npz",
    # ---  DEBUG CONSTANTS
    "STATIC_CAMERA_FEED_WINDOW_TITLE": "Static camera feed",
    "MOVING_CAMERA_FEED_WINDOW_TITLE": "Moving camera feed",
    "WARPED_FRAME_WINDOW_TITLE": "Warped moving frame",
    "LIGHT_DIRECTION_WINDOW_TITLE": "Light direction",
    # ---  INTERACTIVE RELIGHTING CONSTANTS
    "INTERPOLATED_WINDOW_TITLE": "Interpolated Data",
    "INPUT_LIGHT_DIRECTION_WINDOW_TITLE": "Light direction input",
    # ---  COIN DATA FILE NAMES CONSTANTS
    "CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH": "data/coins/static_intrinsics.xml",
    "CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH": "data/coins/moving_intrinsics.xml",
    "COIN_1_ALIGNED_VIDEO_STATIC_PATH": "data/coins/1_static_aligned_video.mov",
    "COIN_1_ALIGNED_VIDEO_MOVING_PATH": "data/coins/1_moving_aligned_video.mp4",
    "COIN_1_PCA_MODEL": "data/coins/1_pca_model.pt",
    "COIN_1_PCA_DATA_FILE_PATH": "data/coins/1_pca_data.npz",
    "COIN_2_ALIGNED_VIDEO_STATIC_PATH": "data/coins/2_static_aligned_video.mov",
    "COIN_2_ALIGNED_VIDEO_MOVING_PATH": "data/coins/2_moving_aligned_video.mp4",
    "COIN_2_PCA_MODEL": "data/coins/2_pca_model.pt",
    "COIN_2_PCA_DATA_FILE_PATH": "data/coins/2_pca_data.npz",
    "COIN_3_ALIGNED_VIDEO_STATIC_PATH": "data/coins/3_static_aligned_video.mov",
    "COIN_3_ALIGNED_VIDEO_MOVING_PATH": "data/coins/3_moving_aligned_video.mp4",
    "COIN_3_PCA_MODEL": "data/coins/3_pca_model.pt",
    "COIN_3_PCA_DATA_FILE_PATH": "data/coins/3_pca_data.npz",
    "COIN_4_ALIGNED_VIDEO_STATIC_PATH": "data/coins/4_static_aligned_video.mov",
    "COIN_4_ALIGNED_VIDEO_MOVING_PATH": "data/coins/4_moving_aligned_video.mp4",
    "COIN_4_PCA_MODEL": "data/coins/4_pca_model.pt",
    "COIN_4_PCA_DATA_FILE_PATH": "data/coins/4_pca_data.npz",
    "COIN_1_EXTRACTED_DATA_FILE_PATH": "data/coins/1_extracted_data.npz",
    "COIN_2_EXTRACTED_DATA_FILE_PATH": "data/coins/2_extracted_data.npz",
    "COIN_3_EXTRACTED_DATA_FILE_PATH": "data/coins/3_extracted_data.npz",
    "COIN_4_EXTRACTED_DATA_FILE_PATH": "data/coins/4_extracted_data.npz",
    "COIN_1_TEST_DATA_FILE_PATH": "data/coins/1_test_data.npz",
    "COIN_2_TEST_DATA_FILE_PATH": "data/coins/2_test_data.npz",
    "COIN_3_TEST_DATA_FILE_PATH": "data/coins/3_test_data.npz",
    "COIN_4_TEST_DATA_FILE_PATH": "data/coins/4_test_data.npz",
    "COIN_1_INTERPOLATED_DATA_RBF_FILE_PATH": "data/coins/1_rbf_interpolated_data.npz",
    "COIN_2_INTERPOLATED_DATA_RBF_FILE_PATH": "data/coins/2_rbf_interpolated_data.npz",
    "COIN_3_INTERPOLATED_DATA_RBF_FILE_PATH": "data/coins/3_rbf_interpolated_data.npz",
    "COIN_4_INTERPOLATED_DATA_RBF_FILE_PATH": "data/coins/4_rbf_interpolated_data.npz",
    "COIN_1_INTERPOLATED_DATA_PTM_FILE_PATH": "data/coins/1_ptm_interpolated_data.npz",
    "COIN_2_INTERPOLATED_DATA_PTM_FILE_PATH": "data/coins/2_ptm_interpolated_data.npz",
    "COIN_3_INTERPOLATED_DATA_PTM_FILE_PATH": "data/coins/3_ptm_interpolated_data.npz",
    "COIN_4_INTERPOLATED_DATA_PTM_FILE_PATH": "data/coins/4_ptm_interpolated_data.npz",
    "COIN_1_INTERPOLATED_DATA_PCA_MODEL_FILE_PATH": "data/coins/1_pca_model_interpolated_data.npz",
    "COIN_2_INTERPOLATED_DATA_PCA_MODEL_FILE_PATH": "data/coins/2_pca_model_interpolated_data.npz",
    "COIN_3_INTERPOLATED_DATA_PCA_MODEL_FILE_PATH": "data/coins/3_pca_model_interpolated_data.npz",
    "COIN_4_INTERPOLATED_DATA_PCA_MODEL_FILE_PATH": "data/coins/4_pca_model_interpolated_data.npz",
    "COIN_1_DATAPOINTS_FILE_PATH": "data/coins/1_datapoints.npz",
    "COIN_2_DATAPOINTS_FILE_PATH": "data/coins/2_datapoints.npz",
    "COIN_3_DATAPOINTS_FILE_PATH": "data/coins/3_datapoints.npz",
    "COIN_4_DATAPOINTS_FILE_PATH": "data/coins/4_datapoints.npz",
    "COIN_1_TEST_DATAPOINTS_FILE_PATH": "data/coins/1_test_datapoints.npz",
    "COIN_2_TEST_DATAPOINTS_FILE_PATH": "data/coins/2_test_datapoints.npz",
    "COIN_3_TEST_DATAPOINTS_FILE_PATH": "data/coins/3_test_datapoints.npz",
    "COIN_4_TEST_DATAPOINTS_FILE_PATH": "data/coins/4_test_datapoints.npz",
    # ---  SYNTH DATA FILE NAMES CONSTANTS
    "SYNTH_LIGHT_DIRECTIONS_FILENAME": "dirs.lp",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_EXTRACTED_DATA_FILE_PATH": "data/synth/single_2_3_extracted_data.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_TEST_DATA_FILE_PATH": "data/synth/single_2_3_test_data.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_PCA_MODEL": "data/synth/single_2_3_pca_model.pt",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_PCA_DATA_FILE_PATH": "data/synth/single_2_3_pca_data.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_DATAPOINTS_FILE_PATH": "data/synth/single_2_3_datapoints.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_TEST_DATAPOINTS_FILE_PATH": "data/synth/single_2_3_test_datapoints.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_INTERPOLATED_DATA_RBF_FILE_PATH": "data/synth/single_2_3_rbf_interpolated_data.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_INTERPOLATED_DATA_PTM_FILE_PATH": "data/synth/single_2_3_ptm_interpolated_data.npz",
    "SYNTH_SINGLE_OBJECT_2_MATERIAL_3_INTERPOLATED_DATA_PCA_MODEL_FILE_PATH": "data/synth/single_2_3_pca_model_interpolated_data.npz",
}
