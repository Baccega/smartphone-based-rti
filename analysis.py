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
from extract_data import extractCoinDataFromVideos, extractSynthDataFromAssets
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


def saveDatapointsToFile(path, data):
    # Get Datapoints
    datapoints = []
    keys = list(data[0][0].keys())
    light_directions_x = [i.split("|")[0] for i in keys]
    light_directions_y = [i.split("|")[1] for i in keys]
    for i in range(len(light_directions_x)):
        datapoints.append((int(light_directions_x[i]), int(light_directions_y[i])))

    # Save to file
    writeDataFile(path, datapoints)


def coinSubMain(interpolation_mode):
    coin = inputCoin()
    debug_mode = inputDebug()

    (
        not_aligned_static_video_path,
        not_aligned_moving_video_path,
        moving_camera_delay,
        static_video_path,
        moving_video_path,
        extracted_data_file_path,
        test_data_file_path,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
        test_datapoints_file_path,
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
    test_data = None

    # Ask to generate aligned videos (if they already exists)
    if inputAlignedVideos(static_video_path, moving_video_path):
        generateAlignedVideo(not_aligned_static_video_path, static_video_path)
        generateAlignedVideo(
            not_aligned_moving_video_path, moving_video_path, moving_camera_delay
        )

    # Ask to extract data (if it already exists)
    if inputExtractedData(extracted_data_file_path):
        # [for each x, y : {"lightDirs_x|lightDirs_y": pixelIntensities}]
        extracted_data, test_data = extractCoinDataFromVideos(
            static_video_path, moving_video_path, debug_mode
        )
        writeDataFile(extracted_data_file_path, extracted_data)
        writeDataFile(test_data_file_path, test_data)
    else:
        extracted_data = loadDataFile(extracted_data_file_path)
        test_data = loadDataFile(test_data_file_path)

    return (
        extracted_data,
        test_data,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
        test_datapoints_file_path,
    )


def synthSubMain(interpolation_mode):
    synth = inputSynth()

    (
        data_folder_path,
        data_light_directions_file_path,
        test_folder_path,
        test_light_directions_file_path,
        extracted_data_file_path,
        test_data_file_path,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
        test_datapoints_file_path,
    ) = getChoosenSynthPaths(synth, interpolation_mode)

    if (not os.path.exists(data_folder_path)) or (
        not os.path.exists(data_light_directions_file_path)
    ):
        raise (Exception("Synth assets not founded!"))

    print("*** Analysis *** \n\tSynthData folder: '{}'".format(data_folder_path))

    extracted_data = None
    test_data = None

    # Ask to extract data (if it already exists)
    if inputExtractedData(extracted_data_file_path):
        # [for each x, y : {"lightDirs_x|lightDirs_y": pixelIntensities}]
        extracted_data, test_data = extractSynthDataFromAssets(
            data_folder_path,
            data_light_directions_file_path,
            test_folder_path,
            test_light_directions_file_path,
        )
        writeDataFile(extracted_data_file_path, extracted_data)
        writeDataFile(test_data_file_path, test_data)
    else:
        extracted_data = loadDataFile(extracted_data_file_path)
        test_data = loadDataFile(test_data_file_path)

    return (
        extracted_data,
        test_data,
        interpolated_data_file_path,
        neural_model_path,
        pca_data_file_path,
        datapoints_file_path,
        test_datapoints_file_path,
    )


def main():
    dataset = inputDataset()
    interpolation_mode = inputInterpolatedMode()

    if dataset == 1:
        (
            extracted_data,
            test_data,
            interpolated_data_file_path,
            neural_model_path,
            pca_data_file_path,
            datapoints_file_path,
            test_datapoints_file_path,
        ) = coinSubMain(interpolation_mode)
    else:
        (
            extracted_data,
            test_data,
            interpolated_data_file_path,
            neural_model_path,
            pca_data_file_path,
            datapoints_file_path,
            test_datapoints_file_path,
        ) = synthSubMain(interpolation_mode)

    # Save lightdirection datapoints (interactive relighting debug)
    saveDatapointsToFile(datapoints_file_path, extracted_data)
    # Save test lightdirection datapoints (interactive relighting debug)
    saveDatapointsToFile(test_datapoints_file_path, test_data)

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

    # Do you want to analyze the results with the test set?


if __name__ == "__main__":
    main()
