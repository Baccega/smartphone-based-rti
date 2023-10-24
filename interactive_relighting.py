import os
import cv2 as cv
import numpy as np
import torch
import math
from constants import constants
from pca_model import PCAModel
from neural_model import NeuralModel

from myIO import inputCoin, inputInterpolatedMode, inputDataset, inputSynth
from utils import (
    boundXY,
    createLightDirectionFrame,
    getChoosenCoinVideosPaths,
    getChoosenSynthPaths,
    loadDataFile,
    getPytorchDevice,
    fromLightDirToIndex,
)
from interpolation import (
    getPCAModelInterpolationFunction,
    getNeuralModelInterpolationFunction,
)

device = getPytorchDevice()
    
torch.manual_seed(42)

dirX = 0.7500
dirY = 0.4330

prevDirX = 0
prevDirY = 0
isDragging = False


def mouse_click(event, x, y, flags, param):
    global dirX, dirY, isDragging

    def click():
        global dirX, dirY
        boundedX, boundedY = boundXY(x, y)
        dirX = boundedX
        dirY = boundedY * -1
        print("Light direction: ({}, {})".format(dirX, dirY))

    # if event == cv.EVENT_LBUTTONDOWN:
    #     click()
    if not isDragging and event == cv.EVENT_LBUTTONDOWN:
        click()
        isDragging = True
    if isDragging and event == cv.EVENT_MOUSEMOVE:
        click()
    if isDragging and event == cv.EVENT_LBUTTONUP:
        click()
        isDragging = False


def mainPreComputed(
    interpolated_data_file_path, datapoints_file_path, test_datapoints_file_path
):
    global dirX, dirY, prevDirX, prevDirY
    frame = np.zeros(
        shape=[
            constants["SQUARE_GRID_DIMENSION"],
            constants["SQUARE_GRID_DIMENSION"],
            3,
        ],
        dtype=np.uint8,
    )

    datapoints = loadDataFile(datapoints_file_path)
    test_datapoints = loadDataFile(test_datapoints_file_path)

    cv.namedWindow(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"])
    lightDirectionFrame = createLightDirectionFrame(
        [dirX, dirY], datapoints, test_datapoints
    )
    cv.setMouseCallback(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"], mouse_click)

    data = loadDataFile(interpolated_data_file_path)

    flag = True
    while flag:
        if prevDirX != dirX or prevDirY != dirY:
            for x in range(constants["SQUARE_GRID_DIMENSION"]):
                for y in range(constants["SQUARE_GRID_DIMENSION"]):
                    frame[x][y] = max(0, min(255, data[dirX][dirY][x][y]))
            lightDirectionFrame = createLightDirectionFrame(
                [dirX, dirY], datapoints, test_datapoints
            )
            prevDirX = dirX
            prevDirY = dirY

        cv.imshow(constants["INTERPOLATED_WINDOW_TITLE"], frame)
        cv.imshow(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"], lightDirectionFrame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            flag = False

    cv.destroyAllWindows()


def mainRealTime(
    interpolation_mode,
    model_path,
    pca_data_file_path,
    datapoints_file_path,
    test_datapoints_file_path,
):
    global dirX, dirY, prevDirX, prevDirY
    frame = np.zeros(
        shape=[
            constants["SQUARE_GRID_DIMENSION"],
            constants["SQUARE_GRID_DIMENSION"],
            3,
        ],
        dtype=np.uint8,
    )

    datapoints = loadDataFile(datapoints_file_path)
    test_datapoints = loadDataFile(test_datapoints_file_path)

    cv.namedWindow(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"])
    lightDirectionFrame = createLightDirectionFrame([dirX, dirY], test_datapoints)
    cv.setMouseCallback(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"], mouse_click)

    if interpolation_mode == 4:
        get_interpolation_function = (
            "PCAModel",
            getPCAModelInterpolationFunction(pca_data_file_path, model_path)[1],
        )
    if interpolation_mode == 6:
        get_interpolation_function = (
            "NeuralModel",
            getNeuralModelInterpolationFunction(model_path)[1],
        )

    (
        interpolation_function_name,
        interpolation_function,
    ) = get_interpolation_function

    print("Neural model: " + model_path)
    print("Interpolation function: " + interpolation_function_name)

    flag = True
    while flag:
        if prevDirX != dirX or prevDirY != dirY:
            outputs = interpolation_function(dirX, dirY, True)

            for x in range(constants["SQUARE_GRID_DIMENSION"]):
                for y in range(constants["SQUARE_GRID_DIMENSION"]):
                    i = y + (x * constants["SQUARE_GRID_DIMENSION"])
                    frame[x][y] = outputs[i]

            lightDirectionFrame = createLightDirectionFrame(
                [dirX, dirY], datapoints, test_datapoints
            )
            prevDirX = dirX
            prevDirY = dirY

        cv.imshow(constants["INTERPOLATED_WINDOW_TITLE"], frame)
        cv.imshow(constants["INPUT_LIGHT_DIRECTION_WINDOW_TITLE"], lightDirectionFrame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            flag = False

    cv.destroyAllWindows()


if __name__ == "__main__":

    dataset = inputDataset()
    interpolation_mode = inputInterpolatedMode()

    if dataset == 1:
        coin = inputCoin()
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            interpolated_data_file_path,
            neural_model_path,
            pca_data_file_path,
            datapoints_file_path,
            test_datapoints_file_path,
        ) = getChoosenCoinVideosPaths(coin, interpolation_mode)
    else:
        synth = inputSynth()

        (
            _,
            _,
            _,
            _,
            _,
            _,
            interpolated_data_file_path,
            neural_model_path,
            pca_data_file_path,
            datapoints_file_path,
            test_datapoints_file_path,
        ) = getChoosenSynthPaths(synth, interpolation_mode)

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
    if interpolation_mode == 6 and (
        (not os.path.exists(neural_model_path))
        or (not os.path.exists(constants["GAUSSIAN_MATRIX_FILE_PATH_XY"]))
        or (not os.path.exists(constants["GAUSSIAN_MATRIX_FILE_PATH_UV"]))
    ):
        raise (
            Exception(
                "You need to run the analysis before the interactive relighting on this coin!"
            )
        )
    if (
        interpolation_mode != 4
        and interpolation_mode != 6
        and (not os.path.exists(interpolated_data_file_path))
    ):
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

    if interpolation_mode == 4 or interpolation_mode == 6:
        mainRealTime(
            interpolation_mode,
            neural_model_path,
            pca_data_file_path,
            datapoints_file_path,
            test_datapoints_file_path,
        )
    else:
        mainPreComputed(
            interpolated_data_file_path, datapoints_file_path, test_datapoints_file_path
        )

    print("All Done!")
