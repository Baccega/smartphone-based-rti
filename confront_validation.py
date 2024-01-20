import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

from myIO import inputSynth
from constants import constants
from interpolation import getNeuralModelInterpolationFunction
from utils import getChoosenSynthPaths, getRtiPaths, get_intermediate_light_directions

N_TO_CHECK = 5

SAVE=True

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
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
        test_datapoints_file_path,
    ) = getRtiPaths(6)

    # First 5 points in validation set
    points = [
        (0.157,0.484,"assets/rti-dataset/val/image0.jpeg"),
        (0.090,0.318,"assets/rti-dataset/val/image1.jpeg"),
        (-0.089,0.204,"assets/rti-dataset/val/image2.jpeg"),
        (0.161,0.884,"assets/rti-dataset/val/image3.jpeg"),
        (0.278,0.722,"assets/rti-dataset/val/image4.jpeg"),
    ]
    
    print("Model path: {}".format(neural_model_path))

    # Get in-between interpolated images
    _, interpolateImage = getNeuralModelInterpolationFunction(neural_model_path)

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
            cv.imwrite(f"image{i}_interpolated.jpeg", frame)
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
