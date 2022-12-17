import numpy as np
import cv2 as cv
from tqdm import tqdm
from constants import constants
from scipy.interpolate import Rbf
from utils import fromIndexToLightDir

N = constants["SQAURE_GRID_DIMENSION"]


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


def SSIM(output, ground_truth):

    return 1


def analyze_data(data, test_data, interpolation_mode=None):

    print("Analyzing data...")

    if interpolation_mode is None:
        return

    get_interpolation_functions = [
        ("LinearRBF", getLinerRBFInterpolationFunction(data)),
        ("PolynomialTextureMaps", getPTMInterpolationFunction(data)),
    ]
    comparison_functions = [("SSIM", SSIM)]

    test_light_directions = list(test_data[0][0].keys())

    for i in range(len(get_interpolation_functions)):
        (
            interpolation_function_name,
            interpolation_function,
        ) = get_interpolation_functions[i]

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

            mean_comparison_value = total_comparison_value / len(test_data)

            print(
                "{} - {}: {} ({} values)".format(
                    interpolation_function_name,
                    comparison_function_name,
                    mean_comparison_value,
                    len(test_data),
                )
            )
