import cv2 as cv
import numpy as np
from features_detector import FeaturesDetector
from utils import loadIntrinsics, getChoosenCoinVideosPaths
import os
from constants import (
    ALIGNED_VIDEO_FPS,
    ANALYSIS_FRAME_SKIP,
    OPTICAL_FLUX_FAILURES_LIMIT,
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

    data = []

    features_detector = FeaturesDetector()

    max_frames = min(
        static_video.get(cv.CAP_PROP_FRAME_COUNT),
        moving_video.get(cv.CAP_PROP_FRAME_COUNT),
    )
    current_frame_count, failures = 0
    flag = True
    while flag:
        is_static_valid, static_frame_distorted = static_video.read()
        is_moving_valid, moving_frame_distorted = moving_video.read()

        if is_static_valid and is_moving_valid:
            static_frame = cv.undistort(static_frame_distorted, K_static, dist_static)
            moving_frame = cv.undistort(moving_frame_distorted, K_moving, dist_moving)

            # TODO Flusso ottico?
            frame_data = features_detector.extrapolateDataFromFrame(
                static_frame=static_frame, moving_frame=moving_frame
            )

            if frame_data:
                failures = 0
                data.append(frame_data)
            else:
                failures += 1
                if failures > OPTICAL_FLUX_FAILURES_LIMIT:
                    features_detector.exceededFailureLimit()

            data.append()

            # Video output during analysis
            cv.imshow(STATIC_CAMERA_FEED_WINDOW_TITLE, static_frame)
            cv.imshow(MOVING_CAMERA_FEED_WINDOW_TITLE, moving_frame)
            cv.waitKey(1)

            # Frame skip
            current_frame_count += ANALYSIS_FRAME_SKIP
            if max_frames > current_frame_count:
                static_video.set(cv.CAP_PROP_POS_FRAMES, current_frame_count)
                moving_video.set(cv.CAP_PROP_POS_FRAMES, current_frame_count)
            else:
                flag = False
        else:
            flag = False

    static_video.release()
    moving_video.release()
    cv.destroyAllWindows()

    return data


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
