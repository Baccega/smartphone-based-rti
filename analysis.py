import cv2 as cv
import numpy as np
from features_detector import findRectanglePatternHomography
from interpolation import interpolate_data
from myIO import (
    inputAlignedVideos,
    inputCoin,
    inputExtractedData,
    inputInterpolatedData,
    inputInterpolatedMode,
)
from utils import (
    createLightDirectionFrame,
    findLightDirection,
    findPixelIntensities,
    fromLightDirToIndex,
    loadDataFile,
    loadIntrinsics,
    getChoosenCoinVideosPaths,
    writeDataFile,
)
import os
from constants import (
    ALIGNED_VIDEO_FPS,
    ANALYSIS_FRAME_SKIP,
    SQAURE_GRID_DIMENSION,
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


def extractDataFromVideos(static_video_path, moving_video_path):
    cv.namedWindow(STATIC_CAMERA_FEED_WINDOW_TITLE)
    cv.namedWindow(MOVING_CAMERA_FEED_WINDOW_TITLE)

    # Open video files
    static_video = cv.VideoCapture(static_video_path)
    moving_video = cv.VideoCapture(moving_video_path)

    # Get camera intrinsics
    K_static, dist_static = loadIntrinsics(CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH)
    K_moving, dist_moving = loadIntrinsics(CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)

    data = [
        [[] * SQAURE_GRID_DIMENSION] * SQAURE_GRID_DIMENSION
        for i in range(SQAURE_GRID_DIMENSION)
    ]

    max_frames = min(
        static_video.get(cv.CAP_PROP_FRAME_COUNT),
        moving_video.get(cv.CAP_PROP_FRAME_COUNT),
    )
    current_frame_count = 0
    flag = True
    while flag:
        is_static_valid, static_frame_distorted = static_video.read()
        is_static_valid = True
        is_moving_valid, moving_frame_distorted = moving_video.read()

        if is_static_valid and is_moving_valid:
            static_frame = cv.undistort(static_frame_distorted, K_static, dist_static)
            moving_frame = cv.undistort(moving_frame_distorted, K_moving, dist_moving)

            static_gray_frame = cv.cvtColor(static_frame, cv.COLOR_BGR2GRAY)
            moving_gray_frame = cv.cvtColor(moving_frame, cv.COLOR_BGR2GRAY)
            (
                static_homography,
                static_corners,
                static_warped_frame,
            ) = findRectanglePatternHomography(static_gray_frame, "static")
            (
                moving_homography,
                moving_corners,
                moving_warped_frame,
            ) = findRectanglePatternHomography(moving_gray_frame, "moving")

            if static_corners is not None and moving_corners is not None:
                light_direction = findLightDirection(
                    static_frame,
                    moving_frame,
                    static_corners,
                    moving_corners,
                )

                lightDirectionFrame = createLightDirectionFrame(fromLightDirToIndex(light_direction))
                cv.imshow("Light direction", lightDirectionFrame)

                pixel_intensities = findPixelIntensities(static_frame)

                # Saving data points to data structure
                if light_direction is not None and pixel_intensities is not None:
                    for x in range(len(pixel_intensities)):
                        for y in range(len(pixel_intensities)):
                            data_point = {
                                "{}|{}".format(
                                    fromLightDirToIndex(light_direction[0]),
                                    fromLightDirToIndex(light_direction[1]),
                                ): pixel_intensities[x][y]
                            }
                            if type(data[x][y]) is dict:
                                data[x][y].update(data_point)
                            else:
                                data[x][y] = data_point

                static_frame = cv.drawContours(
                    static_frame, [static_corners], -1, (0, 0, 255), 3
                )
                # cv.imshow("warped_moving", moving_warped_frame)

            # Video output during analysis
            cv.imshow(STATIC_CAMERA_FEED_WINDOW_TITLE, static_frame)
            cv.imshow(MOVING_CAMERA_FEED_WINDOW_TITLE, moving_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                flag = False
            # if cv.waitKey(1) & 0xFF == ord("p"):  # Pause
            #     isPaused = False
            # if cv.waitKey(1) & 0xFF == ord("c"):  # Continue
            #     isPaused = True

            # Frame skip
            current_frame_count += ANALYSIS_FRAME_SKIP
            if max_frames < current_frame_count:
                flag = False

            if ANALYSIS_FRAME_SKIP > 1:
                static_video.set(cv.CAP_PROP_POS_FRAMES, current_frame_count)
                moving_video.set(cv.CAP_PROP_POS_FRAMES, current_frame_count)

        else:
            flag = False

    static_video.release()
    moving_video.release()
    cv.destroyAllWindows()

    return np.asarray(data)


def main(
    not_aligned_static_video_path,
    not_aligned_moving_video_path,
    moving_camera_delay,
    static_video_path,
    moving_video_path,
    extracted_data_file_path,
    interpolated_data_file_path,
):
    extracted_data = None
    interpolated_data = None

    # Generate aligned videos if not already done
    if inputAlignedVideos(static_video_path, moving_video_path):
        generateAlignedVideo(not_aligned_static_video_path, static_video_path)
        generateAlignedVideo(
            not_aligned_moving_video_path, moving_video_path, moving_camera_delay
        )

    if inputExtractedData(extracted_data_file_path):
        # [for each x, y : {"lightDirs_x|lightDirs_y": pixelIntensities}]
        extracted_data = extractDataFromVideos(static_video_path, moving_video_path)
        writeDataFile(extracted_data_file_path, extracted_data)

    if extracted_data is not None:
        loaded_extracted_data = extracted_data
    else:
        loaded_extracted_data = loadDataFile(extracted_data_file_path)

    key = list(loaded_extracted_data[0][0].keys())[0]
    print("Data {}:".format(key), loaded_extracted_data[0][0][key])

    # Data interpolation
    if inputInterpolatedData(interpolated_data_file_path):
        interpolation_mode = inputInterpolatedMode()
        interpolated_data = interpolate_data(loaded_extracted_data, interpolation_mode)
        writeDataFile(interpolated_data_file_path, interpolated_data)

    # if interpolated_data is not None:
    #     loaded_interpolated_data = interpolated_data
    # else:
    #     loaded_interpolated_data = loadDataFile(interpolated_data_file_path)


if __name__ == "__main__":
    coin = inputCoin()
    (
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
        extracted_data_file_path,
        interpolated_data_file_path,
    ) = getChoosenCoinVideosPaths(coin)

    if (not os.path.exists(CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH)) or (
        not os.path.exists(CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)
    ):
        raise (Exception("You need to run the calibration before the analysis!"))

    print(
        "*** Analysis *** \n\tStatic_Video: '{}' \n\tMoving_Video: '{}'".format(
            not_aligned_static_video_path, not_aligned_moving_video_path
        )
    )

    main(
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
        extracted_data_file_path,
        interpolated_data_file_path,
    )

    print("All Done!")
