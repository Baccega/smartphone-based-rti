import os
import cv2 as cv
import numpy as np
import torch
from constants import constants
from pca_model import NeuralModel

from myIO import inputCoin, inputInterpolatedMode
from utils import (
    boundXY,
    createLightDirectionFrame,
    getChoosenCoinVideosPaths,
    loadDataFile,
    fromIndexToLightDir,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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


def mainPreComputed(interpolated_data_file_path):
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


def mainRealTime(neural_model_path, pca_data_file_path):
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

    pca_data = loadDataFile(pca_data_file_path)
    print("PCA_DATA_LOADED")
    gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])
    print("GAUSSIAN_MATRIX_LOADED")
    print("Neural model: " + neural_model_path)
    model = NeuralModel(gaussian_matrix)
    model.load_state_dict(torch.load(neural_model_path))
    model.eval()

    flag = True
    while flag:
        if prevDirX != dirX or prevDirY != dirY:
            normalizedDirX = fromIndexToLightDir(dirX)
            normalizedDirY = fromIndexToLightDir(dirY)
            for x in range(constants["SQAURE_GRID_DIMENSION"]):
                for y in range(constants["SQAURE_GRID_DIMENSION"]):
                    input = torch.cat(
                        (
                            torch.tensor(pca_data[x][y]),
                            torch.tensor(
                                [
                                    normalizedDirX,
                                    normalizedDirY,
                                ]
                            ),
                        ),
                        dim=-1,
                    )
                    input = input.to(device)
                    output = model(input)
                    frame[x][y] = max(0, min(255, output.item()))
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
        neural_model_path,
        pca_data_file_path,
    ) = getChoosenCoinVideosPaths(coin, interpolation_mode)

    if interpolation_mode == 4 and (
        (not os.path.exists(neural_model_path))
        or (not os.path.exists(pca_data_file_path))
        or (not os.path.exists(constants["GAUSSIAN_MATRIX_FILE_PATH"]))
    ):
        raise (
            Exception(
                "You need to run the analysis before the interactive relighting on this coin!"
            )
        )
    if interpolation_mode != 4 and (not os.path.exists(interpolated_data_file_path)):
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

    mainPreComputed(interpolated_data_file_path)
    mainRealTime(neural_model_path, pca_data_file_path)

    print("All Done!")
