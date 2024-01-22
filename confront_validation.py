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

IS_PCA = False
SAVE = True


def main():
    print("Confront validation")

    # Load neural model trained on 2 images
    if IS_PCA:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            model_path,
            pca_data_file_path,
            _,
            _,
        ) = getRtiPaths(4)
    else:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            model_path,
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

    print("Model path: {}".format(model_path))

    # Get in-between interpolated images
    
    if IS_PCA:
        _, interpolateImage = getPCAModelInterpolationFunction(
            pca_data_file_path, model_path
        )
    else:
        _, interpolateImage = getNeuralModelInterpolationFunction(model_path)

    for i, point in enumerate(points, 1):
        frame = np.zeros(
            shape=[
                constants["SQUARE_GRID_DIMENSION"],
                constants["SQUARE_GRID_DIMENSION"],
                3,
            ],
            dtype=np.uint8,
        )
        outputs = interpolateImage(point[0], point[1], True).cpu().numpy()
        for x in range(constants["SQUARE_GRID_DIMENSION"]):
            for y in range(constants["SQUARE_GRID_DIMENSION"]):
                index = y + (x * constants["SQUARE_GRID_DIMENSION"])
                frame[x][y] = max(0, min(255, outputs[index]))

        if SAVE:
            model = IS_PCA and "pca" or "neural"
            cv.imwrite(f"image{i}_interpolated_{model}.jpeg", frame)
        else:
            cv.imshow(f"Image {i}; {point[0]}, {point[1]}", frame)

        ground_truth_image = cv.imread(point[2])
        ground_truth_image = cv.resize(
            ground_truth_image,
            (
                constants["SQUARE_GRID_DIMENSION"],
                constants["SQUARE_GRID_DIMENSION"],
            ),
        )
        if SAVE:
            cv.imwrite(f"image{i}_ground_truth.jpeg", ground_truth_image)
        else:
            cv.imshow(f"Ground truth {i}; {point[0]}, {point[1]}", ground_truth_image)

    if not SAVE:
        while cv.waitKey(1) != ord("q"):
            pass

    print("Done")


if __name__ == "__main__":
    main()
