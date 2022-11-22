from scipy.interpolate import Rbf
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pca_model import NeuralModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import constants
from utils import loadDataFile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


PCA_ORTHOGONAL_BASES = 8


def interpolate_data(data, mode, neural_model_path, pca_data_file_path):
    print("\t— Interpolating data")

    interpolated_data = np.zeros(
        (
            constants["LIGHT_DIRECTION_WINDOW_SIZE"],
            constants["LIGHT_DIRECTION_WINDOW_SIZE"],
            constants["SQAURE_GRID_DIMENSION"],
            constants["SQAURE_GRID_DIMENSION"],
        )
    )

    if mode == 3:
        pca_data = loadDataFile(pca_data_file_path)
        print("PCA_DATA_LOADED")
        gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])
        print("GAUSSIAN_MATRIX_LOADED")
        print("Neural model: " + neural_model_path)
        model = NeuralModel(gaussian_matrix)
        model.load_state_dict(torch.load(neural_model_path))
        model.eval()

    with torch.no_grad():
        # For every pixel coordinate
        for x in tqdm(range(constants["SQAURE_GRID_DIMENSION"])):
            for y in range(constants["SQAURE_GRID_DIMENSION"]):
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
                    for x1 in range(constants["LIGHT_DIRECTION_WINDOW_SIZE"]):
                        for y1 in range(constants["LIGHT_DIRECTION_WINDOW_SIZE"]):
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
                    for lv in range(constants["LIGHT_DIRECTION_WINDOW_SIZE"]):
                        for lu in range(constants["LIGHT_DIRECTION_WINDOW_SIZE"]):
                            l0 = a_matrix[0] * (lu**2)
                            l1 = a_matrix[1] * (lv**2)
                            l2 = a_matrix[2] * (lu * lv)
                            l3 = a_matrix[3] * lu
                            l4 = a_matrix[4] * lv
                            L = l0 + l1 + l2 + l3 + l4 + a_matrix[5]
                            interpolated_data[lu][lv][x][y] = L
                elif mode == 3:
                    # dataset = InterpolatedPixelsDataset(pca_data)
                    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
                    # with tqdm(dataloader, unit="batch") as tepoch:
                    #     for i, data in enumerate(tepoch):
                    #         x = i % constants["SQAURE_GRID_DIMENSION"]
                    #         y = math.floor(i / constants["SQAURE_GRID_DIMENSION"]) % constants["SQAURE_GRID_DIMENSION"]
                            
                    #         x1 = math.floor(
                    #             i
                    #             / (constants["SQAURE_GRID_DIMENSION"] * constants["SQAURE_GRID_DIMENSION"]) 
                    #         ) % constants["LIGHT_DIRECTION_WINDOW_SIZE"]
                    #         y1 = math.floor(
                    #             i
                    #             / (constants["SQAURE_GRID_DIMENSION"] * constants["SQAURE_GRID_DIMENSION"] * constants["LIGHT_DIRECTION_WINDOW_SIZE"])
                    #         )
                    #         inputs, _ = data
                    #         inputs = inputs.to(device)
                    #         outputs = model(inputs)
                    #         for j in range(64):
                    #             interpolated_data[x1][y1][x][y] = outputs[(i * 64) + j]
                    for x1 in range(constants["LIGHT_DIRECTION_WINDOW_SIZE"]):
                        for y1 in range(constants["LIGHT_DIRECTION_WINDOW_SIZE"]):
                            input = torch.cat(
                                (torch.tensor(pca_data[x][y]), torch.tensor([x1, y1])),
                                dim=-1,
                            )
                            # input.reshape((64, PCA_ORTHOGONAL_BASES + 2))
                            input = input.to(device)
                            output = model(input)[0].item()
                            interpolated_data[x1][y1][x][y] = output

    return interpolated_data
