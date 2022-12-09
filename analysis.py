from interpolation import interpolate_data
from myIO import (
    inputAlignedVideos,
    inputCoin,
    inputDebug,
    inputDataset,
    inputSynth,
    inputExtractedData,
    inputInterpolatedData,
    inputInterpolatedMode,
    inputModelTraining,
)
from utils import (
    loadDataFile,
    getChoosenCoinVideosPaths,
    getChoosenSynthPaths,
    writeDataFile,
    generateGaussianMatrix,
)
from pca_model import train_pca_model
from extract_data import extractDataFromVideos, extractDataFromImages
import os
import torch
from constants import constants
import ffmpeg

# Generate aligned videos
def generateAlignedVideo(not_aligned_video_path, video_path, delay=0):
    print("\t— Generating aligned video: {}".format(video_path))
    ffmpeg.input(not_aligned_video_path, itsoffset=delay).filter(
        "fps", fps=constants["ALIGNED_VIDEO_FPS"]
    ).output(video_path).run()


def getDatapoints(extracted_data):
    datapoints = []
    keys = list(extracted_data[0][0].keys())
    light_directions_x = [i.split("|")[0] for i in keys]
    light_directions_y = [i.split("|")[1] for i in keys]
    for i in range(len(light_directions_x)):
        datapoints.append((int(light_directions_x[i]), int(light_directions_y[i])))
    return datapoints


def coinMain(debug_mode, interpolation_mode):
    coin = inputCoin()

    (
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
        extracted_data_file_path,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
    ) = getChoosenCoinVideosPaths(coin, interpolation_mode)

    if (not os.path.exists(not_aligned_static_video_path)) or (
        not os.path.exists(not_aligned_moving_video_path)
    ):
        raise (Exception("Video assets not founded!"))

    if (not os.path.exists(constants["CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH"])) or (
        not os.path.exists(constants["CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH"])
    ):
        raise (Exception("You need to run the calibration before the analysis!"))
    print(
        "*** Analysis *** \n\tStatic_Video: '{}' \n\tMoving_Video: '{}'".format(
            not_aligned_static_video_path, not_aligned_moving_video_path
        )
    )

    extracted_data = None

    # Ask to generate aligned videos (if they already exists)
    if inputAlignedVideos(static_video_path, moving_video_path):
        generateAlignedVideo(not_aligned_static_video_path, static_video_path)
        generateAlignedVideo(
            not_aligned_moving_video_path, moving_video_path, moving_camera_delay
        )

    # Ask to extract data (if it already exists)
    if inputExtractedData(extracted_data_file_path):
        # [for each x, y : {"lightDirs_x|lightDirs_y": pixelIntensities}]
        extracted_data = extractDataFromVideos(
            static_video_path, moving_video_path, debug_mode
        )
        writeDataFile(extracted_data_file_path, extracted_data)
        # Save lightdirection datapoints (debug)
        datapoints = getDatapoints(extracted_data)
        writeDataFile(datapoints_file_path, datapoints)
        print("Found {} light directions in total".format(len(datapoints)))

    if extracted_data is not None:
        loaded_extracted_data = extracted_data
    else:
        loaded_extracted_data = loadDataFile(extracted_data_file_path)

    return (
        loaded_extracted_data,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
    )


def synthMain(debug_mode, interpolation_mode):
    synth = inputSynth()

    (
        folder_path,
        light_directions_file_path,
        extracted_data_file_path,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
    ) = getChoosenSynthPaths(synth, interpolation_mode)

    if (not os.path.exists(folder_path)) or (
        not os.path.exists(light_directions_file_path)
    ):
        raise (Exception("Synth assets not founded!"))

    print("*** Analysis *** \n\tSynthData folder: '{}'".format(folder_path))

    extracted_data = None

    # Ask to extract data (if it already exists)
    if inputExtractedData(extracted_data_file_path):
        # [for each x, y : {"lightDirs_x|lightDirs_y": pixelIntensities}]
        extracted_data = extractDataFromImages(folder_path, light_directions_file_path, debug_mode)
        writeDataFile(extracted_data_file_path, extracted_data)
        # Save lightdirection datapoints (debug)
        datapoints = getDatapoints(extracted_data)
        writeDataFile(datapoints_file_path, datapoints)
        print("Found {} light directions in total".format(len(datapoints)))

    if extracted_data is not None:
        loaded_extracted_data = extracted_data
    else:
        loaded_extracted_data = loadDataFile(extracted_data_file_path)

    return (
        loaded_extracted_data,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
    )


def main():
    debug_mode = inputDebug()
    dataset = inputDataset()
    interpolation_mode = inputInterpolatedMode()

    if dataset == 1:
        (
            extracted_data,
            interpolated_data_file_path,
            neural_model_path,
            pca_data_file_path,
        ) = coinMain(debug_mode, interpolation_mode)
    else:
        (
            extracted_data,
            interpolated_data_file_path,
            neural_model_path,
            pca_data_file_path,
        ) = synthMain(debug_mode, interpolation_mode)

    interpolated_data = None

    # Train model if necessary
    if (interpolation_mode == 3 or interpolation_mode == 4) and inputModelTraining(
        neural_model_path
    ):
        if not os.path.exists(constants["GAUSSIAN_MATRIX_FILE_PATH"]):
            gaussian_matrix = generateGaussianMatrix(
                0, torch.tensor(constants["PCA_SIGMA"]), constants["PCA_H"]
            )
            writeDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"], gaussian_matrix)
        else:
            gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])

        train_pca_model(
            neural_model_path,
            extracted_data,
            gaussian_matrix,
            pca_data_file_path,
        )

    # Interpolate data from extracted if necessary
    if interpolation_mode != 4 and inputInterpolatedData(interpolated_data_file_path):
        interpolated_data = interpolate_data(
            extracted_data,
            interpolation_mode,
            neural_model_path,
            pca_data_file_path,
        )
        writeDataFile(interpolated_data_file_path, interpolated_data)

    print("All Done! Now you can use the interactive relighting.")


if __name__ == "__main__":
    main()
