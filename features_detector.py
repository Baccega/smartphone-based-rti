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


def findRectanglePatternHomography(gray, choosen_camera, debug_mode):
    """
    Given a gray image, find the rectangle pattern and estimate homography matrix
    """

    # We use findRectanglePatterns and we keep the first (best) result
    polys = findRectanglePatterns(gray, choosen_camera, debug_mode)
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

        # We return the Homography, Corners and the Warped Image
        return M, biggerContour, warped
    return None, None, None


def findRectanglePatterns(gray, choosen_camera, debug_mode):
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
    
    if debug_mode >= 2 and choosen_camera == "moving":
        cv.imshow("threshold", thresh)
        cv.waitKey(1)

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
                sortedCurve = sortCorners(curve)
                score = outerContour(sortedCurve.astype(np.int32), gray)
                area = cv.contourArea(sortedCurve.astype(np.int32))
                polys.append((sortedCurve, score, area))

    # We sort the found rectangles by score descending, using outerContour function
    # The function calculates the mean of the border inside the rectangle: it must be around full black
    # It's safe to take the first entry as a valid pattern if it exists
    polys.sort(key=lambda x: x[1], reverse=False)
    # print(polys)

    return [p[0].astype(np.int32) for p in polys if p[1] < 40 and area > 40000]
