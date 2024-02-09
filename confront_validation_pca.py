import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

from constants import constants
from interpolation import (
    getNeuralModelInterpolationFunction,
    getPCAModelInterpolationFunction,
)
from utils import getRtiPaths, get_intermediate_light_directions

SAVE = True
N_POINTS = 4

def main():
    print("Confront validation")

    # Load neural model trained on 2 images
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        pca_model_path,
        pca_data_file_path,
        _,
        _,
    ) = getRtiPaths(4)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        neural_model_path,
        _,
        _,
        _,
    ) = getRtiPaths(6)

    # First 5 points in validation set
    points = [
        (0.157,0.484,"assets/rti-dataset/val/image0.jpeg"),
        (0.090,0.318,"assets/rti-dataset/val/image1.jpeg"),
        (-0.944,-0.228,"assets/rti-dataset/val/image15.jpeg"),
        (0.161,0.884,"assets/rti-dataset/val/image3.jpeg"),
        (0.726,0.615,"assets/rti-dataset/val/image19.jpeg"),
    ]

    print("Model path: {}".format(pca_model_path))
    print("Model path: {}".format(neural_model_path))

    # Get in-between interpolated images
    
    _, interpolateImagePca = getPCAModelInterpolationFunction(
        pca_data_file_path, pca_model_path
    )
    _, interpolateImageNeural = getNeuralModelInterpolationFunction(neural_model_path)

    for i, point in enumerate(points, 1):
        frame = np.zeros(
            shape=[
                constants["SQUARE_GRID_DIMENSION"],
                constants["SQUARE_GRID_DIMENSION"],
                3,
            ],
            dtype=np.uint8,
        )
        outputsPca = interpolateImagePca(point[0], point[1], True).cpu().numpy()
        outputsNeural = interpolateImageNeural(point[0], point[1], True).cpu().numpy()

        for x in range(constants["SQUARE_GRID_DIMENSION"]):
            for y in range(constants["SQUARE_GRID_DIMENSION"]):
                index = y + (x * constants["SQUARE_GRID_DIMENSION"])
                frame[x][y] = max(0, min(255, outputsPca[index]))
        cv.imwrite(f"results/_pca/image{i}_interpolated_pca.jpg", frame)
        for x in range(constants["SQUARE_GRID_DIMENSION"]):
            for y in range(constants["SQUARE_GRID_DIMENSION"]):
                index = y + (x * constants["SQUARE_GRID_DIMENSION"])
                frame[x][y] = max(0, min(255, outputsNeural[index]))
        cv.imwrite(f"results/_pca/image{i}_interpolated_neural.jpg", frame)

        ground_truth_image = cv.imread(point[2])
        ground_truth_image = cv.resize(
            ground_truth_image,
            (
                constants["SQUARE_GRID_DIMENSION"],
                constants["SQUARE_GRID_DIMENSION"],
            ),
        )
        cv.imwrite(f"results/_pca/image{i}_ground_truth.jpg", ground_truth_image)

    if not SAVE:
        while cv.waitKey(1) != ord("q"):
            pass

    print("Done")


if __name__ == "__main__":
    main()
