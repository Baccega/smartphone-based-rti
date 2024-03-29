import os
import torch
import cv2 as cv
import numpy as np
import math
from constants import constants

SCALE = constants["LIGHT_DIRECTION_WINDOW_SCALE"]


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
    mode_str = "RBF"
    if interpolation_mode == 2:
        mode_str = "PTM"
    elif interpolation_mode == 3 or interpolation_mode == 4:
        mode_str = "PCA"
    elif interpolation_mode == 5 or interpolation_mode == 6:
        mode_str = "NEURAL"
    return (
        constants["COIN_{}_VIDEO_CAMERA_STATIC_PATH".format(coin)],
        constants["COIN_{}_VIDEO_CAMERA_MOVING_PATH".format(coin)],
        constants["FILE_{}_MOVING_CAMERA_DELAY".format(coin)],
        constants["COIN_{}_ALIGNED_VIDEO_STATIC_PATH".format(coin)],
        constants["COIN_{}_ALIGNED_VIDEO_MOVING_PATH".format(coin)],
        constants["COIN_{}_EXTRACTED_DATA_FILE_PATH".format(coin)],
        constants["COIN_{}_TEST_DATA_FILE_PATH".format(coin)],
        constants["COIN_{}_INTERPOLATED_DATA_{}_FILE_PATH".format(coin, mode_str)],
        constants["COIN_{}_{}_MODEL".format(coin, mode_str)],
        constants["COIN_{}_PCA_DATA_FILE_PATH".format(coin)],
        constants["COIN_{}_DATAPOINTS_FILE_PATH".format(coin)],
        constants["COIN_{}_TEST_DATAPOINTS_FILE_PATH".format(coin)],
    )


def getChoosenSynthPaths(synth, interpolation_mode=0):
    """
    Get constants based on the synth object and interpolation mode
    """
    mode_str = "RBF"
    if interpolation_mode == 2:
        mode_str = "PTM"
    elif interpolation_mode == 3 or interpolation_mode == 4:
        mode_str = "PCA"
    elif interpolation_mode == 5 or interpolation_mode == 6:
        mode_str = "NEURAL"

    singleMulti = "Single" if synth[0] == "SINGLE" else "Multi"

    data_folder = "assets/synthRTI/{}/Object{}/material{}/Dome".format(
        singleMulti, synth[1], synth[2]
    )
    test_folder = "assets/synthRTI/{}/Object{}/material{}/Test".format(
        singleMulti, synth[1], synth[2]
    )

    return (
        data_folder,
        "{}/{}".format(data_folder, constants["SYNTH_LIGHT_DIRECTIONS_FILENAME"]),
        test_folder,
        "{}/{}".format(test_folder, constants["SYNTH_LIGHT_DIRECTIONS_FILENAME"]),
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_EXTRACTED_DATA_FILE_PATH".format(
                synth[0], synth[1], synth[2]
            )
        ],
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_TEST_DATA_FILE_PATH".format(
                synth[0], synth[1], synth[2]
            )
        ],
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_INTERPOLATED_DATA_{}_FILE_PATH".format(
                synth[0], synth[1], synth[2], mode_str
            )
        ],
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_{}_MODEL".format(
                synth[0], synth[1], synth[2], mode_str
            )
        ],
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_PCA_DATA_FILE_PATH".format(
                synth[0], synth[1], synth[2]
            )
        ],
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_DATAPOINTS_FILE_PATH".format(
                synth[0], synth[1], synth[2]
            )
        ],
        constants[
            "SYNTH_{}_OBJECT_{}_MATERIAL_{}_TEST_DATAPOINTS_FILE_PATH".format(
                synth[0], synth[1], synth[2]
            )
        ],
    )



def getRtiPaths(interpolation_mode=0):
    """
    Get constants based on the rti object and interpolation mode
    """
    mode_str = "RBF"
    if interpolation_mode == 2:
        mode_str = "PTM"
    elif interpolation_mode == 3 or interpolation_mode == 4:
        mode_str = "PCA"
    elif interpolation_mode == 5 or interpolation_mode == 6:
        mode_str = "NEURAL"

    base_folder = "assets/rti-dataset"
    data_folder_path = "{}/train".format(base_folder)
    data_light_directions_file_path = "{}/light_train".format(base_folder)
    test_folder_path = "{}/val".format(base_folder)
    test_light_directions_file_path = "{}/light_val".format(base_folder)
    extracted_data_file_path = constants["RTI_EXTRACTED_DATA_FILE_PATH"]
    test_data_file_path = constants["RTI_TEST_DATA_FILE_PATH"]
    interpolated_data_file_path = constants["RTI_INTERPOLATED_DATA_{}_FILE_PATH".format(mode_str)]
    model_path = constants["RTI_{}_MODEL".format(mode_str)]
    pca_data_file_path = constants["RTI_PCA_DATA_FILE_PATH"]
    datapoints_file_path = constants["RTI_DATAPOINTS_FILE_PATH"]
    test_datapoints_file_path = constants["RTI_TEST_DATAPOINTS_FILE_PATH"]

    return (
        data_folder_path,
        data_light_directions_file_path,
        test_folder_path,
        test_light_directions_file_path,
        extracted_data_file_path,
        test_data_file_path,
        interpolated_data_file_path,
        model_path,
        pca_data_file_path,
        datapoints_file_path,
        test_datapoints_file_path,
    )


def generateGaussianMatrix(mean, standard_deviation, size):
    first = []
    second = []
    for i in range(size):
        first += [torch.normal(mean, standard_deviation)]
    for i in range(size):
        second += [torch.normal(mean, standard_deviation)]
    return torch.stack([torch.tensor(first), torch.tensor(second)], dim=0).numpy()


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
        (constants["SQUARE_GRID_DIMENSION"], constants["SQUARE_GRID_DIMENSION"]),
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
    center = [constants["SQUARE_GRID_DIMENSION"], constants["SQUARE_GRID_DIMENSION"], 0]
    refSquare = np.array([[0, 400], [400, 400], [400, 0], [0, 0]])

    rotation, translation = computeRt(refSquare, moving_corners)
    o = -rotation.T @ translation

    l = (o - center) / np.linalg.norm(o - center)

    l[0] = l[0] * -1
    # l[1] = l[1] * -1

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


def denormalize(value):
    """
    De-normalize a value from the range [-1, +1] to [0, constants["LIGHT_DIRECTION_WINDOW_SIZE"]].
    """
    return math.floor(
        ((float(value) + 1) / 2) * constants["LIGHT_DIRECTION_WINDOW_SIZE"] * SCALE
    )


def createLightDirectionFrame(light_direction, datapoints=[], test_datapoints=[]):
    """
    Create a frame to show light direction to user
    """
    blank_image = np.zeros(
        shape=[
            constants["LIGHT_DIRECTION_WINDOW_SIZE"] * SCALE,
            constants["LIGHT_DIRECTION_WINDOW_SIZE"] * SCALE,
            3,
        ],
        dtype=np.uint8,
    )

    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] * SCALE / 2)
    cv.line(
        blank_image,
        (half_size, half_size),
        (denormalize(light_direction[0]), denormalize(light_direction[1] * -1)),
        (255, 255, 255),
    )
    cv.circle(
        blank_image,
        (half_size, half_size),
        half_size,
        (255, 255, 255),
    )

    if len(datapoints) > 0:
        for i in range(len(datapoints)):
            blank_image = cv.circle(
                blank_image,
                (denormalize(datapoints[i][0]), denormalize(float(datapoints[i][1]) * -1)),
                radius=0,
                color=(0, 255, 0),
                thickness=-1,
            )
    if len(test_datapoints) > 0:
        for i in range(len(test_datapoints)):
            blank_image = cv.circle(
                blank_image,
                (
                    denormalize(test_datapoints[i][0]),
                    denormalize(float(test_datapoints[i][1]) * -1),
                ),
                radius=0,
                color=(0, 0, 255),
                thickness=-1,
            )

    return blank_image


def boundXY(x, y):
    """
    Force X and Y to be within the light directions bounds
    """
    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] / 2)
    x = math.floor(x / SCALE)
    y = math.floor(y / SCALE)
    if (x - half_size) * (x - half_size) + (y - half_size) * (y - half_size) <= (
        half_size * half_size
    ):
        return (
            (x / constants["LIGHT_DIRECTION_WINDOW_SIZE"]) * 2 - 1,
            (y / constants["LIGHT_DIRECTION_WINDOW_SIZE"]) * 2 - 1,
        )
    else:
        print("OUTSIDE!")
        return (half_size, half_size)


def fromLightDirToIndex(lightDir):
    """
    Transform light direction [-1.0, ..., +1.0] to positive indexes (0, ..., 200)
    """
    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] / 2)
    return int(np.around(lightDir, decimals=2) * half_size) + half_size


def fromIndexToLightDir(index):
    """
    Transform positive indexes (0, ..., 100) to light direction [-1.0, ..., +1.0]
    """
    half_size = int(constants["LIGHT_DIRECTION_WINDOW_SIZE"] / 2)
    return np.around((int(index) - half_size) / half_size, decimals=2)


def normalizeXY(index):
    """
    Transform positive indexes (0, ..., 200) to light direction [-1.0, ..., +1.0]
    """
    half_size = int(constants["SQUARE_GRID_DIMENSION"] / 2)
    return np.around((int(index) - half_size) / half_size, decimals=2)


def writeDataFile(data_file_path, data):
    """
    Write data file to os
    """
    print("Saving data into '{}'...".format(data_file_path))
    np.savez_compressed(data_file_path, data)
    print("Saved!")


def loadDataFile(data_file_path):
    """
    Load data file from os
    """
    print("Loading extracted data file '{}'...".format(data_file_path))
    loaded_data = np.load(data_file_path, allow_pickle=True)["arr_0"]
    print("Loaded!")
    return loaded_data


def getPytorchDevice():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    # Check if MPS is available
    # elif torch.backends.mps.is_available():
    #     print("Using MPS")
    #     device = torch.device("mps:0")
    # Fallback to CPU
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def get_intermediate_light_directions(x1, y1, x2, y2, n):
    # A list to store the interpolated points
    points = []

    # Calculate the differences divided by the number of points + 1
    dx = (x2 - x1) / (n + 1)
    dy = (y2 - y1) / (n + 1)

    # Generate the points and add them to the list
    for i in range(1, n + 1):
        xi = x1 + i * dx
        yi = y1 + i * dy
        points.append((xi, yi))

    return points
