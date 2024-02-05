from scipy.interpolate import Rbf
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, Dataset
from pca_model import PCAModel
from neural_model import NeuralModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import constants
from utils import loadDataFile, fromIndexToLightDir, normalizeXY, getPytorchDevice

device = getPytorchDevice()
torch.manual_seed(42)

N = constants["SQUARE_GRID_DIMENSION"]
M = constants["LIGHT_DIRECTION_WINDOW_SIZE"]

PCA_ORTHOGONAL_BASES = 8


def getLinearRBFInterpolationFunction(data):
    keys = list(data[0][0].keys())
    light_directions_x = [i.split("|")[0] for i in keys]
    light_directions_y = [i.split("|")[1] for i in keys]

    def interpolate(x, y, missing_light_directions):
        pixel_intensities = list(data[x][y].values())
        rbf_interpolation = Rbf(
            light_directions_x,
            light_directions_y,
            pixel_intensities,
            function="linear",
        )

        values = np.empty(len(missing_light_directions))
        for i in range(len(missing_light_directions)):
            light_pair = missing_light_directions[i]
            light_dir_x = light_pair[0]
            light_dir_y = light_pair[1]
            values[i] = rbf_interpolation(light_dir_x, light_dir_y)
        return values

    return interpolate


def getPTMInterpolationFunction(data):
    keys = list(data[0][0].keys())
    light_directions_x = [i.split("|")[0] for i in keys]
    light_directions_y = [i.split("|")[1] for i in keys]

    def interpolate(x, y, missing_light_directions):
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

        values = np.empty(len(missing_light_directions))
        for i in range(len(missing_light_directions)):
            light_pair = missing_light_directions[i]
            lu = int(light_pair[0])
            lv = int(light_pair[1])
            l0 = a_matrix[0] * (lu**2)
            l1 = a_matrix[1] * (lv**2)
            l2 = a_matrix[2] * (lu * lv)
            l3 = a_matrix[3] * lu
            l4 = a_matrix[4] * lv
            L = l0 + l1 + l2 + l3 + l4 + a_matrix[5]
            values[i] = L
        return values

    return interpolate


def getPCAModelInterpolationFunction(pca_data_file_path, pca_model_path):
    pca_data = loadDataFile(pca_data_file_path)
    gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])
    model = PCAModel(gaussian_matrix)
    model = model.to(device)
    model.load_state_dict(torch.load(pca_model_path))
    model.eval()

    def interpolate(x, y, missing_light_directions):
        with torch.no_grad():
            inputs = torch.empty(
                (len(missing_light_directions), 10),
                dtype=torch.float32,
            )

            inputs = inputs.to(device)
            for i in range(len(missing_light_directions)):
                light_pair = missing_light_directions[i]
                light_dir_x = fromIndexToLightDir(light_pair[0])
                light_dir_y = fromIndexToLightDir(light_pair[1])

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
                ).to(device)

            outputs = model(inputs)
            return outputs

    def interpolateImage(dirX, dirY, dir_normalized=False):
        with torch.no_grad():
            inputs = torch.empty(
                (
                    constants["SQUARE_GRID_DIMENSION"]
                    * constants["SQUARE_GRID_DIMENSION"],
                    10,
                ),
                dtype=torch.float32,
            )

            inputs = inputs.to(device)
            if dir_normalized:
                normalizedDirX = dirX
                normalizedDirY = dirY
            else:
                normalizedDirX = fromIndexToLightDir(dirX)
                normalizedDirY = fromIndexToLightDir(dirY)

            for x in range(N):
                for y in range(N):
                    inputs[(x * N) + y] = torch.cat(
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
                    ).to(device)

            outputs = model(inputs)
            return outputs

    return interpolate, interpolateImage


def getNeuralModelInterpolationFunction(model_path):
    gaussian_matrix_xy = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH_XY"])
    gaussian_matrix_uv = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH_UV"])
    model = NeuralModel(gaussian_matrix_xy, gaussian_matrix_uv)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    def interpolate(x, y, missing_light_directions):
        with torch.no_grad():
            inputs = torch.empty(
                (len(missing_light_directions), 4),
                dtype=torch.float32,
            )

            inputs = inputs.to(device)

            normalized_x = normalizeXY(x)
            normalized_y = normalizeXY(y)
            for i in range(len(missing_light_directions)):
                light_pair = missing_light_directions[i]
                light_dir_x = light_pair[0]
                light_dir_y = light_pair[1]

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
                ).to(device)

            outputs = model(inputs)
            return outputs

    def interpolateImage(dirX, dirY, dir_normalized=False):
        with torch.no_grad():
            inputs = torch.empty(
                (
                    constants["SQUARE_GRID_DIMENSION"]
                    * constants["SQUARE_GRID_DIMENSION"],
                    4,
                ),
                dtype=torch.float32,
            )

            inputs = inputs.to(device)

            if dir_normalized:
                normalizedDirX = dirX
                normalizedDirY = dirY
            else:
                normalizedDirX = fromIndexToLightDir(dirX)
                normalizedDirY = fromIndexToLightDir(dirY)

            for x in range(N):
                normalized_x = normalizeXY(x)
                for y in range(N):
                    normalized_y = normalizeXY(y)
                    inputs[(x * N) + y] = torch.cat(
                        (
                            torch.tensor(
                                [
                                    normalized_x,
                                    normalized_y,
                                    normalizedDirX,
                                    normalizedDirY,
                                ]
                            ),
                        ),
                        dim=-1,
                    )

            outputs = model(inputs)
            return outputs

    return interpolate, interpolateImage


def interpolate_data(data, mode, model_path, pca_data_file_path):
    print("\t— Interpolating data")

    interpolated_data = np.zeros(
        (
            M,
            M,
            N,
            N,
        )
    )

    # if mode == 1:
    #     get_interpolation_function = (
    #         "LinearRBF",
    #         getLinearRBFInterpolationFunction(data),
    #     )
    # if mode == 2:
    #     get_interpolation_function = (
    #         "PolynomialTextureMaps",
    #         getPTMInterpolationFunction(data),
    #     )
    # if mode == 3:
    #     get_interpolation_function = (
    #         "PCAModel",
    #         getPCAModelInterpolationFunction(pca_data_file_path, model_path),
    #     )
    # if mode == 5:
    #     get_interpolation_function = (
    #         "NeuralModel",
    #         getNeuralModelInterpolationFunction(model_path),
    #     )

    # (_, interpolation_function) = get_interpolation_function

    # missing_lights_directions = np.empty(
    #     (
    #         M * M,
    #         2,
    #     ),
    # )
    # for i in range(M * M):
    #     u = math.floor(i / N) % N
    #     v = i % N
    #     missing_lights_directions[i] = (u, v)

    # for x in tqdm(range(N)):
    #     for y in range(N):
    #         values = interpolation_function(x, y, missing_lights_directions)

    #         for i in range(M * M):
    #             u = math.floor(i / M) % M
    #             v = i % M
    #             interpolated_data[u][v][x][y] = values[i]

    if mode == 3:
        pca_data = loadDataFile(pca_data_file_path)
        print("PCA_DATA_LOADED")
        gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])
        print("GAUSSIAN_MATRIX_LOADED")
        print("Neural model: " + model_path)
        model = PCAModel(gaussian_matrix)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    with torch.no_grad():
        # For every pixel coordinate
        for x in tqdm(range(N)):
            for y in range(N):
                if mode == 1:
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

                    # For every possible light direction
                    for x1 in range(M):
                        for y1 in range(M):
                            interpolated_data[x1][y1][x][y] = rbf_interpolation(x1, y1)
                elif mode == 2:
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

                    # For every possible light direction
                    for lu in range(M):
                        for lv in range(M):
                            l0 = a_matrix[0] * (lu**2)
                            l1 = a_matrix[1] * (lv**2)
                            l2 = a_matrix[2] * (lu * lv)
                            l3 = a_matrix[3] * lu
                            l4 = a_matrix[4] * lv
                            L = l0 + l1 + l2 + l3 + l4 + a_matrix[5]
                            interpolated_data[lu][lv][x][y] = L
                elif mode == 3:
                    for x1 in range(M):
                        inputs = torch.empty(
                            (M, constants["PCA_H"]),
                            dtype=torch.float32,
                        )
                        normalizedDirX = fromIndexToLightDir(x1)
                        for y1 in range(M):
                            normalizedDirY = fromIndexToLightDir(y1)
                            inputs[y1] = torch.cat(
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

                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        for y1 in range(M):
                            interpolated_data[x1][y1][x][y] = outputs[y1].item()

    return interpolated_data
