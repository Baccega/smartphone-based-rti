import cv2 as cv
import numpy as np
from utils import loadIntrinsics, getChoosenCoinVideosPaths
import os
from constants import (
    ALIGNED_VIDEO_FPS,
    STATIC_CAMERA_FEED_WINDOW_TITLE,
    MOVING_CAMERA_FEED_WINDOW_TITLE,
    CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH,
    CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH,
    CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH,
    CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH,
)
import ffmpeg


def generateAlignedVideo(not_aligned_video_path, video_path, delay=0):
    print("\t— Generating aligned video: {}".format(video_path))
    ffmpeg.input(not_aligned_video_path, itsoffset=delay).filter(
        "fps", fps=ALIGNED_VIDEO_FPS
    ).output(video_path).run()


def getLightDirectionData(static_video_path, moving_video_path):
    cv.namedWindow(STATIC_CAMERA_FEED_WINDOW_TITLE)
    cv.namedWindow(MOVING_CAMERA_FEED_WINDOW_TITLE)

    # Open video files
    static_video = cv.VideoCapture(static_video_path)
    moving_video = cv.VideoCapture(moving_video_path)

    # Get camera intrinsics
    K_static, dist_static = loadIntrinsics(CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH)
    K_moving, dist_moving = loadIntrinsics(CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)

    while True:
        is_static_valid, static_frame_distorted = static_video.read()
        is_moving_valid, moving_frame_distorted = moving_video.read()

        if is_static_valid and is_moving_valid:
            static_frame = cv.undistort(static_frame_distorted, K_static, dist_static)
            moving_frame = cv.undistort(moving_frame_distorted, K_moving, dist_moving)

            cv.imshow(STATIC_CAMERA_FEED_WINDOW_TITLE, static_frame)
            cv.imshow(MOVING_CAMERA_FEED_WINDOW_TITLE, moving_frame)
            cv.waitKey(1)
        else:
            break

    cv.destroyAllWindows()


def main(
    not_aligned_static_video_path,
    not_aligned_moving_video_path,
    moving_camera_delay,
    static_video_path,
    moving_video_path,
):
    print(
        "*** Analysis *** \nStatic_Video: '{}' \nMoving_Video: '{}'".format(
            not_aligned_static_video_path, not_aligned_moving_video_path
        )
    )

    # Generate aligned videos if not already done
    if (not os.path.exists(static_video_path)) or (
        not os.path.exists(moving_video_path)
    ):
        generateAlignedVideo(not_aligned_static_video_path, static_video_path)
        generateAlignedVideo(
            not_aligned_moving_video_path, moving_video_path, moving_camera_delay
        )

    # From moving get lightdir and timestamp
    light_direction_data = getLightDirectionData(static_video_path, moving_video_path)

    # From static get every pixel value for each light direction and timestamp:
    # (timestamp 45.0, lightdir [x,y,0]) -> pixelvalues

    # Data interpolation

    # MOVING:

    # TIME && LIGHTDIR

    # STATIC:

    # RECT -> DIVIDE IN 400 -> VALUE BASED ON TIME


if __name__ == "__main__":
    coin = 1
    (
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
    ) = getChoosenCoinVideosPaths(coin)

    if (not os.path.exists(CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH)) or (
        not os.path.exists(CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)
    ):
        raise (Exception("You need to run the calibration before the analysis!"))

    main(
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
    )
