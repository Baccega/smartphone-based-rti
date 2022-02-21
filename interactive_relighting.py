import os
import cv2 as cv
import numpy as np
from constants import (
    INPUT_LIGHT_DIRECTION_WINDOW_TITLE,
    INTERPOLATED_WINDOW_TITLE,
    SQAURE_GRID_DIMENSION,
)

from myIO import inputCoin
from utils import (
    boundXY,
    createLightDirectionFrame,
    getChoosenCoinVideosPaths,
    loadDataFile,
)

dirX = 69
dirY = 172
prevDirX = None
prevDirY = None


def nothing(x):
    pass


def updateFrameData(data, dirX, dirY):
    newFrame = np.zeros(
        shape=[
            SQAURE_GRID_DIMENSION,
            SQAURE_GRID_DIMENSION,
            3,
        ],
        dtype=np.uint8,
    )
    for x in range(SQAURE_GRID_DIMENSION):
        for y in range(SQAURE_GRID_DIMENSION):
            newFrame[x][y] = data[x][y][dirX][dirY]
    return newFrame


def mouse_click(event, x, y, flags, param):
    global dirX, dirY
    if event == cv.EVENT_LBUTTONDOWN:
        boundedX, boundedY = boundXY(x, y)
        print("Clicked | x: {} ; y: {}".format(boundedX, boundedY))
        dirX = boundedX
        dirY = boundedY


def main(interpolated_data_file_path):
    global dirX, dirY, prevDirX, prevDirY
    frame = np.zeros(
        shape=[
            SQAURE_GRID_DIMENSION,
            SQAURE_GRID_DIMENSION,
            3,
        ],
        dtype=np.uint8,
    )

    cv.namedWindow(INPUT_LIGHT_DIRECTION_WINDOW_TITLE)
    lightDirectionFrame = createLightDirectionFrame([dirX, dirY])
    cv.setMouseCallback(INPUT_LIGHT_DIRECTION_WINDOW_TITLE, mouse_click)

    data = loadDataFile(interpolated_data_file_path)

    flag = True
    while flag:
        if prevDirX != dirX or prevDirY != dirY:
            frame = updateFrameData(data, dirX, dirY)
            lightDirectionFrame = createLightDirectionFrame([dirX, dirY])
            cv.setMouseCallback(INPUT_LIGHT_DIRECTION_WINDOW_TITLE, mouse_click)
            prevDirX = dirX
            prevDirY = dirY

        cv.imshow(INTERPOLATED_WINDOW_TITLE, frame)
        cv.imshow(INPUT_LIGHT_DIRECTION_WINDOW_TITLE, lightDirectionFrame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            flag = False

    cv.destroyAllWindows()


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
