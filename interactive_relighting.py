import os
import cv2 as cv
import numpy as np
from constants import constants

from myIO import inputCoin, inputInterpolatedMode
from utils import (
    boundXY,
    createLightDirectionFrame,
    getChoosenCoinVideosPaths,
    loadDataFile,
)

dirX = 0    
dirY = 0
prevDirX = None
prevDirY = None
isDragging = False


def mouse_click(event, x, y, flags, param):
    global dirX, dirY, isDragging

    def click():
        global dirX, dirY
        boundedX, boundedY = boundXY(x, y)
        dirX = boundedX
        dirY = boundedY

    if not isDragging and event == cv.EVENT_LBUTTONDOWN:
        click()
        isDragging = True
    if isDragging and event == cv.EVENT_MOUSEMOVE:
        click()
    if isDragging and event == cv.EVENT_LBUTTONUP:
        click()
        isDragging = False


def main(interpolated_data_file_path):
    global dirX, dirY, prevDirX, prevDirY
    frame = np.zeros(
        shape=[
            constants["SQAURE_GRID_DIMENSION"],
            constants["SQAURE_GRID_DIMENSION"],
            3,
        ],
        dtype=np.uint8,
    )

    cv.namedWindow(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"])
    lightDirectionFrame = createLightDirectionFrame([dirX, dirY])
    cv.setMouseCallback(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"], mouse_click)

    data = loadDataFile(interpolated_data_file_path)

    flag = True
    while flag:
        if prevDirX != dirX or prevDirY != dirY:
            for x in range(constants["SQAURE_GRID_DIMENSION"]):
                for y in range(constants["SQAURE_GRID_DIMENSION"]):
                    frame[x][y] = max(0, min(255, data[dirX][dirY][x][y]))
            lightDirectionFrame = createLightDirectionFrame([dirX, dirY])
            prevDirX = dirX
            prevDirY = dirY

        cv.imshow(constants["INTERPOLATED_WINDOW_TITLE"], frame)
        cv.imshow(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"], lightDirectionFrame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            flag = False

    cv.destroyAllWindows()


if __name__ == "__main__":
    coin = inputCoin()
    interpolation_mode = inputInterpolatedMode()
    (
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
        extracted_data_file_path,
        interpolated_data_file_path,
        _,
        _,
    ) = getChoosenCoinVideosPaths(coin, interpolation_mode)

    if not os.path.exists(interpolated_data_file_path):
        raise (
            Exception(
                "You need to run the analysis before the interactive relighting on this coin!"
            )
        )

    print(
        "*** Interactive Relighting *** \n\tData: '{}' ".format(
            interpolated_data_file_path
        )
    )

    main(
        interpolated_data_file_path,
    )

    print("All Done!")
