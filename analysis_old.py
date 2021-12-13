import cv2 as cv
import numpy as np
import constants
import features_detector

from utils import (
    loadIntrinsics,
    sortCorners,
    outerContour,
)

STATIC_IMAGE_WINDOW_TITLE = "Static camera feed"
MOVING_IMAGE_WINDOW_TITLE = "Moving camera feed"

APPROX_POLY_ACCURACY = 1
BLUR_KERNEL_SIZE = 37
BLUR_ITERATIONS = 1

OUTER_SQUARE_PERIMETER = 2000
INNER_SQUARE_PERIMETER = 1000


def nothing(x):
    pass


def image_blur(image, blur_kernel_size, iterations=1):
    for k in range(0, iterations):
        image = cv.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    return image


def findRectanglePatterns(firstFrame):
    gray = cv.cvtColor(firstFrame, cv.COLOR_RGBA2GRAY)
    thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 7, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # cv.imshow(STATIC_IMAGE_WINDOW_TITLE, thresh)
    # cv.waitKey(1)

    # Find all the possible contours in thresholded image
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    winSize = (16, 16)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 200, 0.1)
    minContourLength = 30
    polys = []
    for contour in contours:
        if len(contour) >= minContourLength:
            # We approximate a polygon, we are only interested in rectangles (4 points, convex)
            accuracy = cv.getTrackbarPos(
                "APPROX_POLY_ACCURACY", STATIC_IMAGE_WINDOW_TITLE
            )
            epsilon = (accuracy / 100) * cv.arcLength(contour, True)
            curve = cv.approxPolyDP(contour, epsilon, True)
            if len(curve) == 4 and cv.isContourConvex(curve):
                # We use cornerSubPix for floating point refinement
                curve = cv.cornerSubPix(
                    gray, np.float32(curve), winSize, zeroZone, criteria
                )
                sortedCurve = sortCorners(gray, curve)
                score = outerContour(sortedCurve.astype(np.int32), gray)
                polys.append((sortedCurve, score))

    # We sort the found rectangles by score descending, using outerContour function
    # The function calculates the mean of the border inside the rectangle: it must be around full black
    # It's safe to take the first entry as a valid pattern if it exists
    polys.sort(key=lambda x: x[1], reverse=False)
    return [p[0] for p in polys]


def main():
    print("Analysis")

    static_video = cv.VideoCapture(constants.CHOOSEN_VIDEO_STATIC_PATH)
    moving_video = cv.VideoCapture(constants.CHOOSEN_VIDEO_MOVING_PATH)

    # FPS_DIFFERENCE = constants.MOVING_VIDEO_FPS - constants.STATIC_VIDEO_FPS

    # Syncing the static video to the moving video by skipping the
    static_video.set(
        cv.CAP_PROP_POS_FRAMES,
        int(constants.STATIC_VIDEO_FPS * constants.CHOOSEN_VIDEO_DELAY),
    )

    flag = 1

    cv.namedWindow(STATIC_IMAGE_WINDOW_TITLE)
    cv.createTrackbar(
        "APPROX_POLY_ACCURACY",
        STATIC_IMAGE_WINDOW_TITLE,
        APPROX_POLY_ACCURACY,
        15,
        nothing,
    )
    # cv.createTrackbar('BLUR_KERNEL_SIZE',
    #                   STATIC_IMAGE_WINDOW_TITLE, BLUR_KERNEL_SIZE, 40, nothing)
    # cv.createTrackbar('BLUR_ITERATIONS',
    #                   STATIC_IMAGE_WINDOW_TITLE, BLUR_ITERATIONS, 20, nothing)

    # Load our video and read the first frame
    # We will use the first frame to find the reference rectangles and colors for the point cloud

    K_static, dist_static = loadIntrinsics(
        constants.CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH
    )
    K_moving, dist_moving = loadIntrinsics(
        constants.CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH
    )

    while flag:
        is_static_valid, static_frame_distorted = static_video.read()
        is_moving_valid, moving_frame_distorted = moving_video.read()

        if is_static_valid and is_moving_valid:
            static_frame = cv.undistort(static_frame_distorted, K_static, dist_static)
            moving_frame = cv.undistort(moving_frame_distorted, K_moving, dist_moving)
            # outerSquareContours = findOuterSquare(static_frame)
            # cv.drawContours(
            #     static_frame, outerSquareContours, -1, (0, 0, 255), 6)
            # polys = findRectanglePatterns(moving_frame)
            polys = findRectanglePatterns(static_frame)
            # print(polys)

            polys = [p.astype(np.int32) for p in polys]

            if len(polys) > 0:
                rect = polys[0]
                # print(rect[0][0])
                moving_frame = cv.circle(
                    moving_frame, rect[0][0], radius=3, color=(0, 0, 255), thickness=-1
                )
                moving_frame = cv.circle(
                    moving_frame, rect[1][0], radius=3, color=(0, 255, 0), thickness=-1
                )
                moving_frame = cv.circle(
                    moving_frame, rect[2][0], radius=3, color=(255, 0, 0), thickness=-1
                )
                moving_frame = cv.circle(
                    moving_frame,
                    rect[3][0],
                    radius=3,
                    color=(150, 150, 150),
                    thickness=-1,
                )

                # cv.drawContours(moving_frame, [rect], -1, (0, 0, 255))

                markerDestPoints: np.ndarray = np.array(
                    [[[0, 70]], [[0, 0]], [[70, 0]], [[70, 70]]]
                )
                markerHomo = cv.findHomography(sortCorners(markerDestPoints), rect)[0]
                markerPlane = findPlaneFromHomography(markerHomo, K_inv)
            # cv.drawContours(
            #     moving_frame, outerSquareContours, -1, (0, 0, 255), 6)

            # cv.imshow("Altro", static_frame)
            cv.imshow(STATIC_IMAGE_WINDOW_TITLE, moving_frame)
            cv.waitKey(1)

        else:
            flag = 0
    cv.destroyAllWindows()

    # For each frame
    # Find the marker
    # Calculate light direction
    # Find pixel value
    # Save values to data structure

    # Interpolation

    #  TODO
    # allineare video (fps e skip?)

    # flusso ottico

    #  trovare pallina:

    #  findHomography ? (Finds a perspective transformation between two planes.)
    #   prendo i quattro quadrati ai corners e trovo quello con la media di pixels maggiore (quello che ha dei pixel bianchi)
    #
    #   img show [startX:endX, startY:endY]

    #  angolo:
    #  posizione_pixel = intrinsics * view_matrix * posizione_punto
    #  posizione_pixel ->

    #  view_matrix to angle


if __name__ == "__main__":
    main()
