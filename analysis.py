import cv2 as cv
import constants

def main():
    print("Analysis")

    static_video = cv.VideoCapture(constants.CHOOSEN_VIDEO_STATIC_PATH)
    moving_video = cv.VideoCapture(constants.CHOOSEN_VIDEO_MOVING_PATH)

    FPS_DIFFERENCE = constants.MOVING_VIDEO_FPS - constants.STATIC_VIDEO_FPS

    # Syncing the static video to the moving video by skipping the
    static_video.set(cv.CAP_PROP_POS_FRAMES, int(
        constants.STATIC_VIDEO_FPS * constants.CHOOSEN_VIDEO_DELAY))

    flag = 1

    while(flag):
        is_static_valid, static_frame = static_video.read()
        is_moving_valid, moving_frame = moving_video.read()

        if(is_static_valid and is_moving_valid):
            cv.imshow('Static', static_frame)
            cv.imshow('Moving', moving_frame)
            cv.waitKey(1)
        else:
            flag = 0
    cv.destroyAllWindows()

    # For each frame
    # Find the marker
    # Calculate light direction
    # Find pixel value
    # Save values to data structure

    # Interpolation


if __name__ == "__main__":
    main()
