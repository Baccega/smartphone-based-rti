import os
import torch
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


def getChoosenCoinVideosPaths(coin, interpolation_mode=0):
    """
    Get constants based on the coin and interpolation mode
    """
    if interpolation_mode == 0:
        return (
            constants["COIN_{}_VIDEO_CAMERA_STATIC_PATH".format(coin)],
            constants["COIN_{}_VIDEO_CAMERA_MOVING_PATH".format(coin)],
            constants["FILE_{}_MOVING_CAMERA_DELAY".format(coin)],
            constants["COIN_{}_ALIGNED_VIDEO_STATIC_PATH".format(coin)],
            constants["COIN_{}_ALIGNED_VIDEO_MOVING_PATH".format(coin)],
            constants["COIN_{}_EXTRACTED_DATA_FILE_PATH".format(coin)],
            "NO_INTERPOLATION",
            constants["COIN_{}_PCA_MODEL".format(coin)],
        )
    else:
        mode_str = "RBF" if interpolation_mode == 1 else "PTM"
        return (
            constants["COIN_{}_VIDEO_CAMERA_STATIC_PATH".format(coin)],
            constants["COIN_{}_VIDEO_CAMERA_MOVING_PATH".format(coin)],
            constants["FILE_{}_MOVING_CAMERA_DELAY".format(coin)],
            constants["COIN_{}_ALIGNED_VIDEO_STATIC_PATH".format(coin)],
            constants["COIN_{}_ALIGNED_VIDEO_MOVING_PATH".format(coin)],
            constants["COIN_{}_EXTRACTED_DATA_FILE_PATH".format(coin)],
            constants["COIN_{}_INTERPOLATED_DATA_{}_FILE_PATH".format(coin, mode_str)],
            constants["COIN_{}_PCA_MODEL".format(coin)],
        )


def generateGaussianMatrix(mean, standard_deviation, size):
    out = []
    for i in range(size):
        out += [torch.normal(mean, standard_deviation.sqrt())]
    return torch.stack(out, dim=0)


def getProjectedLightsInFourierSpace(light_direction_x, light_direction_y, matrix):
    s = np.dot(np.array(light_direction_x, light_direction_y), matrix)

    return (torch.tensor(np.cos(s)), torch.tensor(np.sin(s)))


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


# Compute the rotation and traslation matrix
def computeRt(objectPoints, imagePoints):
    M, D = getCameraIntrinsics(constants["CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH"])

    objectPoints = np.hstack((objectPoints, np.zeros((objectPoints.shape[0], 1))))
    imagePoints = imagePoints.astype(np.float32)
    success, Rvec, tvec = cv.solvePnP(objectPoints, imagePoints, M, D)
    rodRotMat = cv.Rodrigues(Rvec)[0]
    return rodRotMat, tvec.T[0]


def findLightDirection(moving_corners):
    """
    Get light direction from static_frame and moving frame
    """
    center = [constants["SQAURE_GRID_DIMENSION"], constants["SQAURE_GRID_DIMENSION"], 0]
    refSquare = np.array([[0, 400], [400, 400], [400, 0], [0, 0]])

    rotation, translation = computeRt(refSquare, moving_corners)
    o = -rotation.T @ translation

    l = (o - center) / np.linalg.norm(o - center)

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
    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] / 2)
    return int(np.around(lightDir, decimals=2) * half_size) + half_size


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
