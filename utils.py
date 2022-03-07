import os
import cv2 as cv
import numpy as np
import math
from constants import constants


def outerContour(contour, gray, margin=10):
    """
    Given a contour and an image, returns the mean of the pixels around the contour.
    This is used to detect the rectangle fiducial pattern.
    """
    # We create two masks, one with the poly and one with the poly eroded
    kernel = np.ones((margin, margin), np.uint8)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, contour, 255)
    eroded = cv.erode(mask, kernel)
    mask = cv.bitwise_xor(eroded, mask)

    # We calculate the mean with the two XORed mask
    mean = cv.mean(gray, mask)
    return mean[0]


def sortCorners(corners):
    """
    Sorts an array of corners clockwise
    """
    center = np.sum(corners, axis=0) / len(corners)

    # Returns the point rotation angle in radians from the center
    def rot(point):
        return math.atan2(point[0][0] - center[0][0], point[0][1] - center[0][1])

    sortedCorners = sorted(corners, key=rot, reverse=True)
    return np.roll(sortedCorners, 2, axis=0)


def loadIntrinsics(path=constants["CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH"]):
    """
    Loads camera intrinsics from an xml file. Uses a default path if not provided (intrinsics.xml).
    """
    intrinsics = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()
    dist = intrinsics.getNode("dist").mat()
    return K, dist


def getChoosenCoinVideosPaths(coin, interpolation_mode):
    """
    Get constants based on the coin and interpolation mode
    """
    mode_str = "RBF" if interpolation_mode == 1 else "PTM"
    return (
        constants["COIN_{}_VIDEO_CAMERA_STATIC_PATH".format(coin)],
        constants["COIN_{}_VIDEO_CAMERA_MOVING_PATH".format(coin)],
        constants["FILE_{}_MOVING_CAMERA_DELAY".format(coin)],
        constants["COIN_{}_ALIGNED_VIDEO_STATIC_PATH".format(coin)],
        constants["COIN_{}_ALIGNED_VIDEO_MOVING_PATH".format(coin)],
        constants["COIN_{}_EXTRACTED_DATA_FILE_PATH".format(coin)],
        constants["COIN_{}_INTERPOLATED_DATA_{}_FILE_PATH".format(coin, mode_str)],
    )


def findPixelIntensities(static_frame):
    """
    Get pixel intensities from static_frame frame using an ad-hoc roi
    """
    roi = static_frame[
        720 : 720 + 460,
        320 : 320 + 460,
    ]
    roi_full_size = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    roi = cv.resize(
        roi_full_size,
        (constants["SQAURE_GRID_DIMENSION"], constants["SQAURE_GRID_DIMENSION"]),
    )

    return roi[:, :, 2]


def findLightDirection(static_frame, moving_frame, static_corners, moving_corners):
    """
    Get light direction from static_frame and moving frame
    """
    moving_frame = cv.cvtColor(moving_frame, cv.COLOR_BGR2GRAY)
    image_size = moving_frame.shape[::-1]

    M, D = getCameraIntrinsics(constants["CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH"])
    z_axis = 1
    flags = cv.CALIB_USE_INTRINSIC_GUESS

    points_3d = np.float32(
        [
            (static_corners[point][0][0], static_corners[point][0][1], z_axis)
            for point in range(0, len(static_corners))
        ]
    )
    points_2d = np.float32(
        [
            (moving_corners[point][0][0], moving_corners[point][0][1])
            for point in range(0, len(moving_corners))
        ]
    )

    # perform a camera calibration to get R and T
    (_, _, _, r_vectors, t_vectors) = cv.calibrateCamera(
        [points_3d], [points_2d], image_size, cameraMatrix=M, distCoeffs=D, flags=flags
    )
    R = cv.Rodrigues(r_vectors[0])[0]
    T = t_vectors[0]

    pose = -np.matrix(R).T * np.matrix(T)
    pose = np.array(pose).flatten()

    h2, w2 = int(static_frame.shape[0] / 2), int(static_frame.shape[1] / 2)
    p = (h2, w2, 0)
    l = (pose - p) / np.linalg.norm(pose - p)

    # -1 ≤ l[0] ≤ +1
    # -1 ≤ l[1] ≤ +1
    return l


def getCameraIntrinsics(calibration_file_path):
    """
    Get camera intrinsic matrix and distorsion
    """
    Kfile = cv.FileStorage(calibration_file_path, cv.FILE_STORAGE_READ)
    intrinsics_matrix = Kfile.getNode("K").mat()
    distortion_matrix = Kfile.getNode("distortion").mat()

    return intrinsics_matrix, distortion_matrix


def createLightDirectionFrame(light_direction):
    """
    Create a frame to show light direction to user
    """
    blank_image = np.zeros(
        shape=[
            constants["LIGHT_DIRECTION_WINDOW_SIZE"],
            constants["LIGHT_DIRECTION_WINDOW_SIZE"],
            3,
        ],
        dtype=np.uint8,
    )

    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] / 2)

    cv.line(
        blank_image,
        (half_size, half_size),
        (light_direction[0], light_direction[1]),
        (255, 255, 255),
    )
    cv.circle(
        blank_image,
        (half_size, half_size),
        half_size,
        (255, 255, 255),
    )
    return blank_image


def boundXY(x, y):
    """
    Force X and Y to be within the light directions bounds
    """
    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] / 2)
    if (x - half_size) * (x - half_size) + (y - half_size) * (y - half_size) <= (
        half_size * half_size
    ):
        return (x, y)
    else:
        print("OUTSIDE!")
        return (half_size, half_size)


def fromLightDirToIndex(lightDir):
    """
    Transform light direction [-1.0, ..., +1.0] to positive indexes (0, ..., 200)
    """
    return int(np.around(lightDir, decimals=2) * 100) + 100


def writeDataFile(extracted_data_file_path, extracted_data):
    """
    Write data file to os
    """
    print("Saving extracted data into '{}'...".format(extracted_data_file_path))
    np.savez_compressed(extracted_data_file_path, extracted_data)
    print("Saved!")


def loadDataFile(extracted_data_file_path):
    """
    Load data file from os
    """
    print("Loading extracted data file '{}'...".format(extracted_data_file_path))
    loaded_data = np.load(extracted_data_file_path, allow_pickle=True)["arr_0"]
    print("Loaded!")
    return loaded_data
