from scipy.interpolate import Rbf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import constants


def interpolate_data(data, mode):
    print("\t— Interpolating data")

    interpolated_data = np.zeros(
        (
            constants["SQAURE_GRID_DIMENSION"],
            constants["SQAURE_GRID_DIMENSION"],
            200,
            200,
        )
    )

    for x in tqdm(range(constants["SQAURE_GRID_DIMENSION"])):
        for y in range(constants["SQAURE_GRID_DIMENSION"]):
            keys = list(data[x][y].keys())
            light_directions_x = [i.split("|")[0] for i in keys]
            light_directions_y = [i.split("|")[1] for i in keys]
            pixel_intensities = list(data[x][y].values())

            if mode == 1:
                rbf_interpolation = Rbf(
                    light_directions_x,
                    light_directions_y,
                    pixel_intensities,
                    function="linear",
                )

                for x1 in range(200):
                    for y1 in range(200):
                        interpolated_data[x1][y1][x][y] = rbf_interpolation(x1, y1)
            else:
                # compute l_matrix and L_matrix first
                l_matrix = []
                L_matrix = []
                for i in range(len(pixel_intensities)):
                    # compute the PTM row for l_matrix
                    lu, lv = int(light_directions_x[i]), int(light_directions_y[i])
                    row = (lu ** 2, lv ** 2, lu * lv, lu, lv, 1.0)
                    l_matrix.append(row)
                    # add Luminance to L_matrix
                    L_matrix.append(pixel_intensities[i])

                l_matrix = np.array(l_matrix)
                L_matrix = np.array(L_matrix)

                # now we'll fine the a_matrix solving A * a = L
                # solve with svd decomposition
                u, s, v = np.linalg.svd(l_matrix)
                c = np.dot(u.T, L_matrix)
                w = np.divide(c[: len(s)], s)
                a_matrix = np.dot(v.T, w)

                for lv in range(200):
                    for lu in range(200):
                        # the tuple (lu, lv) means (x, y)
                        l0 = a_matrix[0] * (lu ** 2)
                        l1 = a_matrix[1] * (lv ** 2)
                        l2 = a_matrix[2] * (lu * lv)
                        l3 = a_matrix[3] * lu
                        l4 = a_matrix[4] * lv
                        L = l0 + l1 + l2 + l3 + l4 + a_matrix[5]
                        interpolated_data[lu][lv][x][y] = L

    return interpolated_data
