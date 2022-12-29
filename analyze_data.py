import numpy as np
import kornia
import torch
import cv2 as cv
from pca_model import PCAModel
from neural_model import NeuralModel
from tqdm import tqdm
from constants import constants
from scipy.interpolate import Rbf
from utils import fromIndexToLightDir, loadDataFile, normalizeXY

N = constants["SQAURE_GRID_DIMENSION"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def getLinerRBFInterpolationFunction(data):
    def interpolate(x, y, light_directions):
        keys = list(data[x][y].keys())
        light_directions_x = [i.split("|")[0] for i in keys]
        light_directions_y = [i.split("|")[1] for i in keys]

        pixel_intensities = list(data[x][y].values())
        rbf_interpolation = Rbf(
            light_directions_x,
            light_directions_y,
            pixel_intensities,
            function="linear",
        )

        values = []
        for light_direction_str in light_directions:
            light_dir_x = light_direction_str.split("|")[0]
            light_dir_y = light_direction_str.split("|")[1]
            values.append(rbf_interpolation(light_dir_x, light_dir_y))
        return values

    return interpolate


def getPTMInterpolationFunction(data):
    def interpolate(x, y, light_directions):
        keys = list(data[x][y].keys())
        light_directions_x = [i.split("|")[0] for i in keys]
        light_directions_y = [i.split("|")[1] for i in keys]
        pixel_intensities = list(data[x][y].values())
        # Compute light projection matrix
        light_projection_matrix = []
        luminance_vector = []
        for i in range(len(pixel_intensities)):
            lu, lv = int(light_directions_x[i]), int(light_directions_y[i])
            row = (lu**2, lv**2, lu * lv, lu, lv, 1.0)
            light_projection_matrix.append(row)
        light_projection_matrix = np.array(light_projection_matrix)

        # Compute luminance vector
        for i in range(len(pixel_intensities)):
            luminance_vector.append(pixel_intensities[i])
        luminance_vector = np.array(luminance_vector)

        # Solve this equation for a (vector of coefficients):
        # light_projection_matrix * a = luminance_vector
        # Solving using Singular Value Decomposition (SVD)
        u, s, v = np.linalg.svd(light_projection_matrix)
        c = np.dot(u.T, luminance_vector)
        w = np.divide(c[: len(s)], s)
        a_matrix = np.dot(v.T, w)

        values = []
        for light_direction_str in light_directions:
            lu = int(light_direction_str.split("|")[0])
            lv = int(light_direction_str.split("|")[1])
            l0 = a_matrix[0] * (lu**2)
            l1 = a_matrix[1] * (lv**2)
            l2 = a_matrix[2] * (lu * lv)
            l3 = a_matrix[3] * lu
            l4 = a_matrix[4] * lv
            L = l0 + l1 + l2 + l3 + l4 + a_matrix[5]
            values.append(L)
        return values

    return interpolate


def getPCAModelInterpolationFunction(pca_data_file_path, pca_model_path):
    pca_data = loadDataFile(pca_data_file_path)
    gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])
    model = PCAModel(gaussian_matrix)
    model.load_state_dict(torch.load(pca_model_path))
    model.eval()

    def interpolate(x, y, light_directions):
        inputs = torch.empty(
            (len(light_directions), 10),
            dtype=torch.float64,
        )

        inputs = inputs.to(device)
        for i in range(len(light_directions)):
            light_direction_str = light_directions[i]
            light_dir_x = fromIndexToLightDir(light_direction_str.split("|")[0])
            light_dir_y = fromIndexToLightDir(light_direction_str.split("|")[1])

            inputs[i] = torch.cat(
                (
                    torch.tensor(pca_data[x][y]),
                    torch.tensor(
                        [
                            light_dir_x,
                            light_dir_y,
                        ]
                    ),
                ),
                dim=-1,
            )

        outputs = model(inputs)
        return outputs

    return interpolate

def getNeuralModelInterpolationFunction(neural_model_path):
    gaussian_matrix_xy = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH_XY"])
    gaussian_matrix_uv = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH_UV"])
    model = NeuralModel(gaussian_matrix_xy, gaussian_matrix_uv)
    model.load_state_dict(torch.load(neural_model_path))
    model.eval()

    def interpolate(x, y, light_directions):
        inputs = torch.empty(
            (len(light_directions), 4),
            dtype=torch.float64,
        )

        inputs = inputs.to(device)

        normalized_x = normalizeXY(x)
        normalized_y = normalizeXY(y)
        for i in range(len(light_directions)):
            light_direction_str = light_directions[i]
            light_dir_x = fromIndexToLightDir(light_direction_str.split("|")[0])
            light_dir_y = fromIndexToLightDir(light_direction_str.split("|")[1])

            inputs[i] = torch.cat(
                (
                    torch.tensor(
                        [
                            normalized_x,
                            normalized_y,
                            light_dir_x,
                            light_dir_y,
                        ]
                    ),
                ),
                dim=-1,
            )

        outputs = model(inputs)
        return outputs

    return interpolate


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
        get_interpolation_functions = (
            "LinearRBF",
            getLinerRBFInterpolationFunction(data),
        )
    if interpolation_mode == 2:
        get_interpolation_functions = (
            "PolynomialTextureMaps",
            getPTMInterpolationFunction(data),
        )
    if interpolation_mode == 3 or interpolation_mode == 4:
        get_interpolation_functions = (
            "PCAModel",
            getPCAModelInterpolationFunction(pca_data_file_path, model_path),
        )
    if interpolation_mode == 5 or interpolation_mode == 6:
        get_interpolation_functions = (
            "NeuralModel",
            getNeuralModelInterpolationFunction(model_path),
        )

    comparison_functions = [("SSIM", SSIM), ("PSNR", PSNR), ("L1", L1)]

    test_light_directions = list(test_data[0][0].keys())

    (
        interpolation_function_name,
        interpolation_function,
    ) = get_interpolation_functions

    print("{}:".format(interpolation_function_name))

    outputs = np.zeros((len(test_light_directions), N, N), dtype=np.uint8)
    ground_truths = np.zeros((len(test_light_directions), N, N), dtype=np.uint8)

    for x in tqdm(range(N)):
        for y in range(N):
            values = interpolation_function(x, y, test_light_directions)

            for i in range(len(test_light_directions)):
                outputs[i][x][y] = values[i]

            count = 0
            for test_light_direction_str in test_light_directions:
                ground_truths[count][x][y] = test_data[x][y][test_light_direction_str]
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
