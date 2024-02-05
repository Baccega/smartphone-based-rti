import numpy as np
import kornia
import torch
import cv2 as cv
from tqdm import tqdm
from constants import constants
from interpolation import (
    getLinearRBFInterpolationFunction,
    getPTMInterpolationFunction,
    getPCAModelInterpolationFunction,
    getNeuralModelInterpolationFunction,
)
from utils import getPytorchDevice

N = constants["SQUARE_GRID_DIMENSION"]

device = getPytorchDevice()

torch.manual_seed(42)


def SSIM(output, ground_truth):
    tensor1 = kornia.utils.image_to_tensor(output).float()
    tensor2 = kornia.utils.image_to_tensor(ground_truth).float()
    # Add Batch dimension
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)

    values = kornia.metrics.ssim(
        tensor1, tensor2, constants["SSIM_GAUSSIAN_KERNEL_SIZE"], 255.0
    )
    return values.sum() / (N * N)


def PSNR(output, ground_truth):
    tensor1 = kornia.utils.image_to_tensor(output).float()
    tensor2 = kornia.utils.image_to_tensor(ground_truth).float()
    return kornia.metrics.psnr(tensor1, tensor2, 255.0)


def L1(output, ground_truth):
    tensor1 = kornia.utils.image_to_tensor(output).float()
    tensor2 = kornia.utils.image_to_tensor(ground_truth).float()
    loss = torch.nn.L1Loss()
    return loss(tensor1, tensor2)


def analyze_data(
    data, test_data, interpolation_mode, pca_data_file_path="", model_path=""
):
    print("Analyzing data...")

    if interpolation_mode == 1:
        get_interpolation_function = (
            "LinearRBF",
            getLinearRBFInterpolationFunction(data),
        )
    if interpolation_mode == 2:
        get_interpolation_function = (
            "PolynomialTextureMaps",
            getPTMInterpolationFunction(data),
        )
    if interpolation_mode == 3 or interpolation_mode == 4:
        get_interpolation_function = (
            "PCAModel",
            getPCAModelInterpolationFunction(pca_data_file_path, model_path)[0],
        )
    if interpolation_mode == 5 or interpolation_mode == 6:
        get_interpolation_function = (
            "NeuralModel",
            getNeuralModelInterpolationFunction(model_path)[0],
        )

    comparison_functions = [("SSIM", SSIM), ("PSNR", PSNR), ("L1", L1)]

    light_keys = list(test_data[0][0].keys())
    test_light_directions = np.zeros((len(light_keys), 2))
    for i in range(len(light_keys)):
        splitted = light_keys[i].split("|")
        test_light_directions[i] = (splitted[0], splitted[1])

    (
        interpolation_function_name,
        interpolation_function,
    ) = get_interpolation_function

    print("{}:".format(interpolation_function_name))

    outputs = np.zeros((len(light_keys), N, N), dtype=np.uint8)
    ground_truths = np.zeros((len(light_keys), N, N), dtype=np.uint8)

    for x in tqdm(range(N)):
        for y in range(N):
            values = interpolation_function(x, y, test_light_directions)

            for i in range(len(light_keys)):
                outputs[i][x][y] = values[i]

            count = 0
            for light_pair in test_light_directions:
                ground_truths[count][x][y] = test_data[x][y][
                    "{}|{}".format(float(light_pair[0]), float(light_pair[1]))
                ]
                count += 1
            count = 0

    # cv.imshow("ground truths", ground_truths[0])
    # cv.imshow("outputs", outputs[0])
    # cv.waitKey(0)

    for j in range(len(comparison_functions)):
        total_comparison_value = 0.0
        comparison_function_name, comparison_function = comparison_functions[j]

        for idx in range(len(test_light_directions)):
            total_comparison_value += comparison_function(
                outputs[idx], ground_truths[idx]
            )

        mean_comparison_value = total_comparison_value / len(test_light_directions)

        print(
            "{} - {}: {} ({} values)".format(
                interpolation_function_name,
                comparison_function_name,
                mean_comparison_value,
                len(test_light_directions),
            )
        )
