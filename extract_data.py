import cv2 as cv
import numpy as np
import random
from features_detector import findRectanglePatternHomography
from myIO import (
    debugCorners,
)
from utils import (
    createLightDirectionFrame,
    findLightDirection,
    findPixelIntensities,
    fromLightDirToIndex,
    loadDataFile,
    loadIntrinsics,
)
from constants import constants


def selectTestLights(n, data):
    test_data = [
        [[] * constants["SQUARE_GRID_DIMENSION"]] * constants["SQUARE_GRID_DIMENSION"]
        for i in range(constants["SQUARE_GRID_DIMENSION"])
    ]

    # For each point to extract
    for i in range(n):
        lights = list(data[0][0].keys())
        n_lights = len(lights)
        choosen = random.randint(0, n_lights - 1)
        choosen_light_str = lights[choosen]
        choosen_light_x = choosen_light_str.split("|")[0]
        choosen_light_y = choosen_light_str.split("|")[1]

        for x in range(constants["SQUARE_GRID_DIMENSION"]):
            for y in range(constants["SQUARE_GRID_DIMENSION"]):
                data_point = {choosen_light_str: data[x][y][choosen_light_str]}
                # If data[x][y] exists: update
                if type(test_data[x][y]) is dict:
                    test_data[x][y].update(data_point)
                # Else: create it
                else:
                    test_data[x][y] = data_point

                # Remove choosen data (TODO: surroundings)
                del data[x][y][choosen_light_str]

    return data, np.asarray(test_data)


# Extract light direction and pixel intensities data from videos
def extractCoinDataFromVideos(static_video_path, moving_video_path, debug_mode):
    cv.namedWindow(constants["STATIC_CAMERA_FEED_WINDOW_TITLE"])
    cv.namedWindow(constants["MOVING_CAMERA_FEED_WINDOW_TITLE"])

    # Open video files
    static_video = cv.VideoCapture(static_video_path)
    moving_video = cv.VideoCapture(moving_video_path)

    # Get camera intrinsics
    K_static, dist_static = loadIntrinsics(
        constants["CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH"]
    )
    K_moving, dist_moving = loadIntrinsics(
        constants["CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH"]
    )

    # Prepare data structure
    data = [
        [[] * constants["SQUARE_GRID_DIMENSION"]] * constants["SQUARE_GRID_DIMENSION"]
        for i in range(constants["SQUARE_GRID_DIMENSION"])
    ]

    max_frames = min(
        static_video.get(cv.CAP_PROP_FRAME_COUNT),
        moving_video.get(cv.CAP_PROP_FRAME_COUNT),
    )
    current_frame_count = 0
    flag = True
    while flag:
        is_static_valid, static_frame_distorted = static_video.read()
        is_moving_valid, moving_frame_distorted = moving_video.read()

        if is_static_valid and is_moving_valid:
            static_frame = cv.undistort(static_frame_distorted, K_static, dist_static)
            moving_frame = cv.undistort(moving_frame_distorted, K_moving, dist_moving)

            # Get marker's homography and corners from video frames
            static_gray_frame = cv.cvtColor(static_frame, cv.COLOR_BGR2GRAY)
            moving_gray_frame = cv.cvtColor(moving_frame, cv.COLOR_BGR2GRAY)
            (
                static_homography,
                static_corners,
                static_warped_frame,
            ) = findRectanglePatternHomography(static_gray_frame, "static", debug_mode)
            (
                moving_homography,
                moving_corners,
                moving_warped_frame,
            ) = findRectanglePatternHomography(moving_gray_frame, "moving", debug_mode)

            if static_corners is not None and moving_corners is not None:
                # Get light direction from frames and corners
                light_direction = findLightDirection(
                    moving_corners,
                )

                if debug_mode >= 1:
                    # Create debug light direciton window
                    lightDirectionFrame = createLightDirectionFrame(
                        (
                            fromLightDirToIndex(light_direction[0]),
                            fromLightDirToIndex(light_direction[1]),
                        )
                    )
                    cv.imshow("Light direction", lightDirectionFrame)

                # Get pixel intensities from static frame
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
                            # If data[x][y] exists: update
                            if type(data[x][y]) is dict:
                                data[x][y].update(data_point)
                            # Else: create it
                            else:
                                data[x][y] = data_point

                if debug_mode >= 1:
                    # Show marker's contours
                    static_frame = cv.drawContours(
                        static_frame, [static_corners], -1, (0, 0, 255), 3
                    )
                if debug_mode >= 2:
                    # Show moving warped frame
                    cv.imshow(
                        constants["WARPED_FRAME_WINDOW_TITLE"], moving_warped_frame
                    )
                    # Show frames corners
                    static_frame = debugCorners(static_frame, static_corners)
                    moving_frame = debugCorners(moving_frame, moving_corners)

            if debug_mode >= 1:
                # Video output during analysis
                cv.imshow(constants["STATIC_CAMERA_FEED_WINDOW_TITLE"], static_frame)
                cv.imshow(constants["MOVING_CAMERA_FEED_WINDOW_TITLE"], moving_frame)

                # Quit by pressing 'q'
                if cv.waitKey(1) & 0xFF == ord("q"):
                    flag = False

            # Frame skip
            current_frame_count += constants["ANALYSIS_FRAME_SKIP"]
            if max_frames < current_frame_count:
                flag = False

            if constants["ANALYSIS_FRAME_SKIP"] > 1:
                static_video.set(cv.CAP_PROP_POS_FRAMES, current_frame_count)
                moving_video.set(cv.CAP_PROP_POS_FRAMES, current_frame_count)

        else:
            flag = False

    static_video.release()
    moving_video.release()
    cv.destroyAllWindows()

    data = np.asarray(data)

    # Select test data from extracted
    data, test_data = selectTestLights(constants["COINS_TEST_N_LIGHTS"], data)

    return data, test_data


def extractSynthDataFromFolder(folder_path, light_directions_file_path):
    data = [
        [[] * constants["SQUARE_GRID_DIMENSION"]] * constants["SQUARE_GRID_DIMENSION"]
        for i in range(constants["SQUARE_GRID_DIMENSION"])
    ]

    light_directions_file = open(light_directions_file_path, "r")
    count = 0

    for unstripped_line in light_directions_file.readlines():
        line = unstripped_line.strip()

        # Skip first line of file
        if count != 0:
            splitted_line = line.split(" ")
            image_path = "{}/{}".format(folder_path, splitted_line[0])
            light_dir_x = float(splitted_line[1])
            light_dir_y = float(splitted_line[2]) * -1
            # light_dir_z = fromLightDirToIndex(splitted_line[3])

            full_res_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            image = cv.resize(
                full_res_image,
                (
                    constants["SQUARE_GRID_DIMENSION"],
                    constants["SQUARE_GRID_DIMENSION"],
                ),
            )

            for x in range(constants["SQUARE_GRID_DIMENSION"]):
                for y in range(constants["SQUARE_GRID_DIMENSION"]):
                    key = "{}|{}".format(
                        light_dir_x,
                        light_dir_y,
                    )
                    # If data[x][y] exists: update
                    if type(data[x][y]) is dict:
                        data[x][y][key] = image[x][y]
                    # Else: create it
                    else:
                        data[x][y] = {
                            key: image[x][y]
                        }
        count += 1

    return np.asarray(data)


def extractSynthDataFromAssets(
    data_folder_path,
    data_light_directions_file_path,
    test_folder_path,
    test_light_directions_file_path,
):
    print("Extracting data from images...")
    data = extractSynthDataFromFolder(data_folder_path, data_light_directions_file_path)

    print("Extracting test data from images...")
    test_data = extractSynthDataFromFolder(
        test_folder_path, test_light_directions_file_path
    )

    return data, test_data
