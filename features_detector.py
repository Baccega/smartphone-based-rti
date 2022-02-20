import cv2 as cv
import numpy as np

from utils import outerContour, sortCorners


ACCURACY = 1

# Based on marker's rectangle sizes
WARPED_W = 2000
WARPED_H = 2000

ROI1 = 125
ROI2 = ROI1 + 50
ROI3 = 125
ROI4 = ROI3 + 50


# def extrapolateLightDirectionFromFrame(frame):
#     return False


# def extrapolatePixelIntensitiesFromFrame(frame):
#     # print("pixel intensities")
#     return True


def checkBlankArea(warped):
    """
    Check the mean of the area expected to be an empty chess.
    To align the chessboard image find a minimum of this value (white area).
    """
    roi = warped[ROI1:ROI2, ROI3:ROI4]
    mean = cv.mean(roi)
    return mean[0]


def findRectanglePatternHomography(gray, choosen_camera):
    """
    Given a gray image, find the rectangle pattern and estimate homography matrix
    """

    # We use findRectanglePatterns and we keep the first (best) result
    polys = findRectanglePatterns(gray, choosen_camera)
    if len(polys) > 0:
        biggerContour = polys[0]

        # We try estimating the homography and warping
        destPoints: np.ndarray = np.array(
            [[[0, WARPED_H]], [[0, 0]], [[WARPED_W, 0]], [[WARPED_W, WARPED_H]]]
        )
        M = cv.findHomography(biggerContour, destPoints)[0]
        warped = cv.warpPerspective(gray, M, (WARPED_W, WARPED_H))

        # ...but it may be rotated, so we need to rectify our pattern.
        # To do this we iterate through all the possible 90 degrees rotations to find the one with a blank tile (upper right).
        # We have the checkBlankArea function that returns the color of our check area, we simply find the minimum.

        currentContour = biggerContour
        currMax = checkBlankArea(warped)
        for i in range(4):
            # Find homography and warped image with that rotation
            currentContour = np.roll(currentContour, shift=1, axis=0)
            M2 = cv.findHomography(currentContour, destPoints)[0]
            rotated = cv.warpPerspective(gray, M2, (WARPED_W, WARPED_H))
            rotatedScore = checkBlankArea(rotated)
            if rotatedScore > currMax:
                biggerContour = currentContour
                M = M2
                warped = rotated
                currMax = rotatedScore

        # cv.imshow("warped", warped)
        # cv.waitKey(1)

        # We return the Homography, Corners and the Warped Image
        return M, biggerContour, warped
    return None, None, None


def findRectanglePatterns(gray, choosen_camera):
    if choosen_camera == "static":
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 17, 4
        )
    else:
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 7, 2
        )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # if choosen_camera == "moving":
    #     cv.imshow("threshold", thresh)
    #     cv.waitKey(1)

    # Find all the possible contours in thresholded image
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    winSize = (16, 16)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 200, 0.1)
    minContourLength = 30
    polys = []
    for contour in contours:
        if len(contour) >= minContourLength:
            epsilon = 0.01 * cv.arcLength(contour, True)
            curve = cv.approxPolyDP(contour, epsilon, True)
            if len(curve) == 4 and cv.isContourConvex(curve):
                # We use cornerSubPix for floating point refinement
                curve = cv.cornerSubPix(
                    gray, np.float32(curve), winSize, zeroZone, criteria
                )
                sortedCurve = sortCorners(gray, curve)
                score = outerContour(sortedCurve.astype(np.int32), gray)
                area = cv.contourArea(sortedCurve.astype(np.int32))
                polys.append((sortedCurve, score, area))

    # We sort the found rectangles by score descending, using outerContour function
    # The function calculates the mean of the border inside the rectangle: it must be around full black
    # It's safe to take the first entry as a valid pattern if it exists
    polys.sort(key=lambda x: x[1], reverse=False)
    # print(polys)

    return [p[0].astype(np.int32) for p in polys if p[1] < 40 and area > 40000]


def extrapolateLightDirectionFromFrame_old(frame):
    status = np.array([1])
    corners = None
    light_direction = None

    polys = findRectanglePatterns(frame)

    if len(polys) > 0:
        corners = polys[0]
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # corner_only_frame = np.zeros(gray_frame.shape, np.uint8)
        cv.drawContours(
            frame,
            polys,
            -1,
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )

        destPoints: np.ndarray = np.array(
            [
                [[0, 512]],
                [[0, 0]],
                [[512, 0]],
                [[512, 512]],
            ]
        )
        homography = cv.findHomography(corners, destPoints)[0]
        print(homography)
        warped = cv.warpPerspective(frame, homography, (512, 512))
        # cv.imshow("warped", warped)
        # cv.waitKey(1)

        # def findRectanglePatternHomography(gray):

        # cv.imshow("test", corner_only_frame)
        # cv.waitKey(1)

        # if status.all() and corners is not None and len(corners) == 4:
        # first_corner = corners[0][0]
        # second_corner = corners[1][0]
        # third_corner = corners[2][0]
        # fourth_corner = corners[3][0]

        #     # Setting up previous_corners for optical flow
        #     self.previous_corners = corners
        #     self.previous_frame = frame

        # frame = cv.circle(
        #     frame,
        #     (int(first_corner[0]), int(first_corner[1])),
        #     radius=3,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )
        # frame = cv.circle(
        #     frame,
        #     (int(second_corner[0]), int(second_corner[1])),
        #     radius=3,
        #     color=(0, 255, 0),
        #     thickness=-1,
        # )
        # frame = cv.circle(
        #     frame,
        #     (int(third_corner[0]), int(third_corner[1])),
        #     radius=3,
        #     color=(255, 0, 0),
        #     thickness=-1,
        # )
        # frame = cv.circle(
        #     frame,
        #     (int(fourth_corner[0]), int(fourth_corner[1])),
        #     radius=3,
        #     color=(150, 150, 150),
        #     thickness=-1,
        # )
        return True
    #     # cv.drawContours(frame, [corners], -1, (0, 0, 255))

    #     corners = (
    #         first_corner,
    #         second_corner,
    #         third_corner,
    #         fourth_corner,
    #     )

    # Light direction
    # light_direction = findLightDirection(
    #     moving_frame, static_corners, moving_corners
    # )

    # return light_direction
    return False
