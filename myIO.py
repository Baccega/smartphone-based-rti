import os
import cv2 as cv


def safeIntInput(message, min_value, max_value):
    value = None
    flag = True
    while flag:
        value = int(input(message))
        if value >= min_value and value <= max_value:
            flag = False
        else:
            print("\nInvalid selection: {}".format(value))
    return value


def safeBoolInput(message):
    value = None
    flag = True
    while flag:
        value = str(input(message))
        if value == "y" or value == "n":
            flag = False
        else:
            print("\nInvalid selection: {}".format(value))
    return value == "y"


def askIfFilesExist(files, message):
    doesFilesExists = True
    flags = [os.path.exists(file) for file in files]
    for flag in flags:
        if not flag:
            doesFilesExists = False
    if doesFilesExists:
        return safeBoolInput(message)
    return True


def inputAnalysis():
    return safeBoolInput(
        "Do you want to compare the extracted data with test set? (y/n): "
    )

def inputCoin():
    return safeIntInput(
        "Select coin (1. Danish 5 krone, 2. British 50 pence, 3. Swiss 1/2 franc, 4. Czechs 20 korun): ",
        1,
        4,
    )

def inputSynth():
    # return ("SINGLE", 2, 3)
    return ("MULTI", 2, 3)
    # return safeIntInput(
    #     "Select coin (1. Danish 5 krone, 2. British 50 pence, 3. Swiss 1/2 franc, 4. Czechs 20 korun): ",
    #     1,
    #     4,
    # )


def inputDebug():
    return safeIntInput(
        "Select debug mode (0. No debug, 1. Minimal debug, 2. Full debug): ",
        0,
        2,
    )

def inputDataset():
    return safeIntInput(
        "Select dataset (1. CoinDataset, 2. SynthDataset): ",
        1,
        2,
    )


def inputAlignedVideos(path1, path2):
    return askIfFilesExist(
        [path1, path2], "Aligned videos found! Do you wish to re-generate them? (y/n): "
    )


def inputExtractedData(path1):
    return askIfFilesExist(
        [path1], "Extracted data found! Do you wish to re-extract it? (y/n): "
    )


def inputInterpolatedData(path1):
    return askIfFilesExist(
        [path1], "Intepolated data found! Do you wish to re-interpolate it? (y/n): "
    )

def inputModelTraining(path1):
    return askIfFilesExist(
        [path1], "Neural model found! Do you wish to re-train it? (y/n): "
    )


def inputInterpolatedMode():
    return safeIntInput(
        "Select interpolation method (1. Linear RBF, 2. Polinomial Texture Maps, 3. PCA neural model (pre-computed), 4. PCA neural model (real-time), 5. Neural model (pre-computed), 6. Neural model (real-time) ): ",
        1,
        6,
    )


def debugCorners(frame, corners):
    first_corner = corners[0][0]
    second_corner = corners[1][0]
    third_corner = corners[2][0]
    fourth_corner = corners[3][0]

    frame = cv.circle(
        frame,
        (int(first_corner[0]), int(first_corner[1])),
        radius=3,
        color=(0, 0, 255),
        thickness=-1,
    )
    frame = cv.circle(
        frame,
        (int(second_corner[0]), int(second_corner[1])),
        radius=3,
        color=(0, 255, 0),
        thickness=-1,
    )
    frame = cv.circle(
        frame,
        (int(third_corner[0]), int(third_corner[1])),
        radius=3,
        color=(255, 0, 0),
        thickness=-1,
    )
    frame = cv.circle(
        frame,
        (int(fourth_corner[0]), int(fourth_corner[1])),
        radius=3,
        color=(150, 150, 150),
        thickness=-1,
    )
    cv.drawContours(frame, [corners], -1, (0, 0, 255))
    return frame
