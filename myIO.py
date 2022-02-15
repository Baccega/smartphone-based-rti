import os


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


def inputCoin():
    return safeIntInput(
        "Select coin (1. Danish 5 krone, 2. British 50 pence, 3. Swiss 1/2 franc, 4. Czechs 20 korun): ",
        1,
        4,
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


def inputInterpolatedMode():
    return safeIntInput(
        "Select interpolation method (1. Linear RBF, 2. Polinomial Texture Maps): ",
        1,
        2,
    )
