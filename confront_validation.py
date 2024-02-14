import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

from constants import constants
from interpolation import (
    getNeuralModelInterpolationFunction
)
from utils import getRtiPaths, getChoosenSynthPaths, get_intermediate_light_directions

IS_SYNTH = True
SAVE = True


def main():
    print("Confront validation")

    # Load neural model trained on 2 images
    if IS_SYNTH == True:
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
        ) = getChoosenSynthPaths(6)
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
        
    if IS_SYNTH == True:
        points = [
            (0.9397,0.000,"assets/synthRTI/Single/Object2/material3/image02.jpg"),
            (-0.9397,-0.000,"assets/synthRTI/Single/Object2/material3/image06.jpg"),
            (0.2932,-0.7077,"assets/synthRTI/Single/Object2/material3/image10.jpg"),
            (-0.2932,0.7077,"assets/synthRTI/Single/Object2/material3/image12.jpg"),
            (-0.4619,0.1913,"assets/synthRTI/Single/Object2/material3/image15.jpg") 
        ]
    else:
        points = [
            (0.157,0.484,"assets/rti-dataset/val/image0.jpeg"),
            (0.090,0.318,"assets/rti-dataset/val/image1.jpeg"),
            (-0.944,-0.228,"assets/rti-dataset/val/image15.jpeg"),
            (0.161,0.884,"assets/rti-dataset/val/image3.jpeg"),
            (0.726,0.615,"assets/rti-dataset/val/image19.jpeg"),
        ]

    print("Model path: {}".format(model_path))

    # Get in-between interpolated images
    
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
            cv.imwrite(f"image{i}_interpolated_neural.jpeg", frame)
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
