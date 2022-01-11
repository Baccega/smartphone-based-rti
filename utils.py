import cv2
import numpy as np
import math
import constants


def outerContour(contour, gray, margin=10):
    """
    Given a contour and an image, returns the mean of the pixels around the contour.
    This is used to detect the rectangle fiducial pattern.
    """
    # We create two masks, one with the poly and one with the poly eroded
    kernel = np.ones((margin, margin), np.uint8)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    eroded = cv2.erode(mask, kernel)
    mask = cv2.bitwise_xor(eroded, mask)

    # We calculate the mean with the two XORed mask
    mean = cv2.mean(gray, mask)
    return mean[0]


def sortCorners(image, corners):
    """
    Sorts an array of corners based on the circle of the marker.
    """
    center = np.sum(corners, axis=0) / len(corners)

    # Returns the point rotation angle in radians from the center
    def rot(point):
        return math.atan2(point[0][0] - center[0][0], point[0][1] - center[0][1])

    sortedCorners = sorted(corners, key=rot, reverse=True)
    return np.roll(sortedCorners, 2, axis=0)


def loadIntrinsics(path=constants.CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH):
    """
    Loads camera intrinsics from an xml file. Uses a default path if not provided (intrinsics.xml).
    """
    intrinsics = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()
    dist = intrinsics.getNode("dist").mat()
    return K, dist


def getChoosenCoinVideosPaths(coin):
    if coin == 1:
        return (
            constants.COIN_1_VIDEO_CAMERA_STATIC_PATH,
            constants.COIN_1_VIDEO_CAMERA_MOVING_PATH,
            constants.FILE_1_MOVING_CAMERA_DELAY,
            constants.COIN_1_ALIGNED_VIDEO_STATIC_PATH,
            constants.COIN_1_ALIGNED_VIDEO_MOVING_PATH,
        )
    elif coin == 2:
        return (
            constants.COIN_2_VIDEO_CAMERA_STATIC_PATH,
            constants.COIN_2_VIDEO_CAMERA_MOVING_PATH,
            constants.FILE_2_MOVING_CAMERA_DELAY,
            constants.COIN_2_ALIGNED_VIDEO_STATIC_PATH,
            constants.COIN_2_ALIGNED_VIDEO_MOVING_PATH,
        )
    elif coin == 3:
        return (
            constants.COIN_3_VIDEO_CAMERA_STATIC_PATH,
            constants.COIN_3_VIDEO_CAMERA_MOVING_PATH,
            constants.FILE_3_MOVING_CAMERA_DELAY,
            constants.COIN_3_ALIGNED_VIDEO_STATIC_PATH,
            constants.COIN_3_ALIGNED_VIDEO_MOVING_PATH,
        )
    elif coin == 4:
        return (
            constants.COIN_4_VIDEO_CAMERA_STATIC_PATH,
            constants.COIN_4_VIDEO_CAMERA_MOVING_PATH,
            constants.FILE_4_MOVING_CAMERA_DELAY,
            constants.COIN_4_ALIGNED_VIDEO_STATIC_PATH,
            constants.COIN_4_ALIGNED_VIDEO_MOVING_PATH,
        )
    else:
        raise Exception("Invaild coin selected")


def findLightDirection(moving_frame, static_corners, moving_corners):
    moving_frame = cv2.cvtColor(moving_frame, cv2.COLOR_BGR2GRAY)
    image_size = moving_frame.shape[::-1]

    M, D = getCameraIntrinsics(constants.CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)
    z_axis = 1
    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    points_3d = np.float32(
        [
            (static_corners[point][0], static_corners[point][1], z_axis)
            for point in range(0, len(static_corners))
        ]
    )
    points_2d = np.float32(
        [
            (moving_corners[point][0], moving_corners[point][1])
            for point in range(0, len(moving_corners))
        ]
    )

    # perform a camera calibration to get R and T
    (ret, matrix, distortion, r_vecs, t_vecs) = cv2.calibrateCamera(
        [points_3d], [points_2d], image_size, cameraMatrix=M, distCoeffs=D, flags=flags
    )
    R = cv2.Rodrigues(r_vecs[0])[0]
    T = t_vecs[0]

    light_direction = -np.matrix(R).T * np.matrix(T)
    light_direction = np.array(light_direction).flatten()

    print(light_direction)
    return light_direction.all()
    # return None


def getCameraIntrinsics(calibration_file_path):
    import os

    """
    Get camera intrinsic matrix and distorsion
    :param calibration_file_path: file path to intrinsics file
    """
    if not os.path.isfile(calibration_file_path):
        raise Exception("intrinsics file not found!")
    else:
        # Read intrinsics to file
        Kfile = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
        matrix = Kfile.getNode("K").mat()
        distortion = Kfile.getNode("distortion").mat()

    return matrix, distortion
