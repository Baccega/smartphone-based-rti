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

N_IN_BETWEEN = 5


def main():
    print("Vgg loss test")

    # Define the VGG model
    vgg_model = models.vgg16(weights="IMAGENET1K_V1").features
    vgg_model.eval()

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
    # TODO: Get these points dynamically
    x1, y1 = 0.277, 0.141
    # x2, y2 = 0.8529, -0.4924
    x2, y2 = -0.883, 0.008

    ground_truth_path = "assets/rti-dataset/train/image0.jpeg"
    
    # synth = inputSynth()
    # (
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     _,
    #     interpolated_data_file_path,
    #     neural_model_path,
    #     pca_data_file_path,
    #     datapoints_file_path,
    #     test_datapoints_file_path,
    # ) = getChoosenSynthPaths(synth, 6)
    # # TODO: Get these points dynamically
    # x1, y1 = 0.7500, 0.4330
    # # x2, y2 = 0.8529, -0.4924
    # x2, y2 = 0.6113, 0.1986

    # ground_truth_path = "assets/synthRTI/Single/Object2/material3/Dome/image20.jpg"
    
    print("Model path: {}".format(neural_model_path))

    # Get ground truth image
    ground_truth_image = cv.imread(ground_truth_path)
    ground_truth_image = cv.resize(
        ground_truth_image,
        (
            constants["SQUARE_GRID_DIMENSION"],
            constants["SQUARE_GRID_DIMENSION"],
        ),
    )

    points = get_intermediate_light_directions(x1, y1, x2, y2, N_IN_BETWEEN)

    for i, point in enumerate(points, 1):
        print(f"Point {i}: x = {point[0]}, y = {point[1]}")

    # Get 10/20 in between directions
    print(N_IN_BETWEEN)

    # Get in-between interpolated images
    _, interpolateImage = getNeuralModelInterpolationFunction(neural_model_path)

    # Prepare vgg loss
    losses = []
    frame = np.zeros(
        shape=[
            constants["SQUARE_GRID_DIMENSION"],
            constants["SQUARE_GRID_DIMENSION"],
            3,
        ],
        dtype=np.uint8,
    )
    outputs = interpolateImage(x1, y1, True).cpu().numpy()
    for x in range(constants["SQUARE_GRID_DIMENSION"]):
        for y in range(constants["SQUARE_GRID_DIMENSION"]):
            index = y + (x * constants["SQUARE_GRID_DIMENSION"])
            frame[x][y] = max(0, min(255, outputs[index]))
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
            ),
        ]
    )
    target_image = Image.fromarray(frame)
    target_image = transform(target_image)
    target_features = vgg_model(target_image.unsqueeze(0))

    cv.imshow(f"Target image", frame)

    outputs = interpolateImage(x2, y2, True).cpu().numpy()
    for x in range(constants["SQUARE_GRID_DIMENSION"]):
        for y in range(constants["SQUARE_GRID_DIMENSION"]):
            index = y + (x * constants["SQUARE_GRID_DIMENSION"])
            frame[x][y] = max(0, min(255, outputs[index]))

    # Add loss from target image 2
    generated_image = Image.fromarray(frame)
    generated_image = transform(generated_image)
    generated_features = vgg_model(generated_image.unsqueeze(0))
    last_loss = nn.functional.mse_loss(target_features, generated_features).item()

    cv.imshow(f"Target image 2", frame)

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

        # Show image
        cv.imshow(f"Image {i}; {point[0]}, {point[1]}", frame)

        # Get vgg loss for each image
        generated_image = Image.fromarray(frame)
        generated_image = transform(generated_image)
        generated_features = vgg_model(generated_image.unsqueeze(0))

        # Calculate the mean squared error between the features
        loss = nn.functional.mse_loss(target_features, generated_features)
        losses.append(loss.item())

    losses.append(last_loss)

    # Plot vgg loss for each image
    # plt.plot(losses)
    # plt.xlabel("Image")
    # plt.ylabel("Loss")
    # plt.title("VGG Perceptual Loss")

    # x_labels = [str(i) for i in range(1, len(losses))]
    # x_labels.append("(2)")
    # plt.xticks(ticks=range(len(losses)), labels=x_labels)

    # plt.show()

    # Show ground truth image
    cv.imshow("Ground truth", ground_truth_image)

    while cv.waitKey(1) != ord("q"):
        pass

    print("Done")

if __name__ == "__main__":
    main()
