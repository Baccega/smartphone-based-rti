import cv2 as cv
import numpy as np
import utils
import os
import constants
import ffmpeg


def generateAlignedVideo(not_aligned_video_path, video_path, delay=0):
    print("\t— Generating aligned video: {}".format(video_path))
    ffmpeg.input(not_aligned_video_path, itsoffset=delay).filter(
        "fps", fps=constants.ALIGNED_VIDEO_FPS
    ).output(video_path).run()


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
    ) = utils.getChoosenCoinVideosPaths(coin)

    if (not os.path.exists(constants.CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH)) or (
        not os.path.exists(constants.CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)
    ):
        raise (Exception("You need to run the calibration before the analysis!"))

    main(
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
    )
