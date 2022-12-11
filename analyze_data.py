import numpy as np
import cv2 as cv
from tqdm import tqdm
from constants import constants
from scipy.interpolate import Rbf
from utils import fromIndexToLightDir


def getLinerRBFInterpolationFunction(data):
    def interpolate(x, y, light_directions):
        keys = list(data[x][y].keys())
        light_directions_x = [fromIndexToLightDir(i.split("|")[0]) for i in keys]
        light_directions_y = [fromIndexToLightDir(i.split("|")[1]) for i in keys]
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


def SSIM(output, ground_truth):

    return 1


def analyze_data(data, test_data, interpolation_mode=None):

    print("Analyzing data...")

    if interpolation_mode is None:
        return

    get_interpolation_functions = [
        ("LinearRBF", getLinerRBFInterpolationFunction(data))
    ]
    comparison_functions = [("SSIM", SSIM)]

    test_light_directions = list(test_data[0][0].keys())

    for i in range(len(get_interpolation_functions)):
        (
            interpolation_function_name,
            interpolation_function,
        ) = get_interpolation_functions[i]

        outputs = np.zeros(
            (
                constants["SQAURE_GRID_DIMENSION"],
                constants["SQAURE_GRID_DIMENSION"],
                len(test_light_directions),
            )
        )
        ground_truths = np.zeros(
            (
                constants["SQAURE_GRID_DIMENSION"],
                constants["SQAURE_GRID_DIMENSION"],
                len(test_light_directions),
            )
        )
        for x in tqdm(range(constants["SQAURE_GRID_DIMENSION"])):
            for y in range(constants["SQAURE_GRID_DIMENSION"]):
                outputs[x][y] = interpolation_function(x, y, test_light_directions)

                count = 0
                for value in test_data[x][y].values():
                    ground_truths[x][y][count] = value
                    count += 1
        outputs = outputs.reshape(
            (
                len(test_light_directions),
                constants["SQAURE_GRID_DIMENSION"],
                constants["SQAURE_GRID_DIMENSION"],
            )
        )
        ground_truths = ground_truths.reshape(
            (
                len(test_light_directions),
                constants["SQAURE_GRID_DIMENSION"],
                constants["SQAURE_GRID_DIMENSION"],
            )
        )

        cv.imshow("test2", outputs[0])
        cv.waitKey(0)

        for j in range(len(comparison_functions)):
            total_comparison_value = 0.0
            comparison_function_name, comparison_function = comparison_functions[j]

            print(
                "{} - {}".format(interpolation_function_name, comparison_function_name)
            )

            
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
