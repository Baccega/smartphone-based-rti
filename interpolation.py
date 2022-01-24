from scipy.interpolate import Rbf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import SQAURE_GRID_DIMENSION


def interpolate_data(data, mode):
    print("\t— Interpolating data")

    interpolated_data = [
        [[[[] * 200] * 200] * SQAURE_GRID_DIMENSION] * SQAURE_GRID_DIMENSION
        for i in range(SQAURE_GRID_DIMENSION)
    ]
    # plt.scatter(x=data[0][0][0], y=data[0][0][1], c=data[0][0][2])
    # plt.xlabel("lx")
    # plt.ylabel("ly")
    # plt.show()

    for x in tqdm(range(SQAURE_GRID_DIMENSION)):
        for y in tqdm(range(SQAURE_GRID_DIMENSION)):
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
                        interpolated_data[x][y][x1][y1] = rbf_interpolation(x1, y1)
            else:
                print("OUCH")
                # _interpolate_RBF()
    return interpolated_data
