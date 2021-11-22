import numpy as np
import cv2 as cv

CHOOSEN_COIN = 1
SQAURE_GRID_DIMENSION = 400 # It will be a 400x400 square grid inside the marker

STATIC_VIDEO_FPS = 29.97
MOVING_VIDEO_FPS = 30.01

COIN_1_VIDEO_CAMERA_STATIC_PATH = 'data/cam1-static/coin1.mov'
COIN_1_VIDEO_CAMERA_MOVING_PATH = 'data/cam2-moving_light/coin1.mp4'
COIN_2_VIDEO_CAMERA_STATIC_PATH = 'data/cam1-static/coin2.mov'
COIN_2_VIDEO_CAMERA_MOVING_PATH = 'data/cam2-moving_light/coin2.mp4'
COIN_3_VIDEO_CAMERA_STATIC_PATH = 'data/cam1-static/coin3.mov'
COIN_3_VIDEO_CAMERA_MOVING_PATH = 'data/cam2-moving_light/coin3.mp4'
COIN_4_VIDEO_CAMERA_STATIC_PATH = 'data/cam1-static/coin4.mov'
COIN_4_VIDEO_CAMERA_MOVING_PATH = 'data/cam2-moving_light/coin4.mp4'

FILE_1_STATIC_CAMERA_DELAY = 2.724  # [seconds] (static) 3.609 - 0.885 (moving)
FILE_2_STATIC_CAMERA_DELAY = 2.024  # [seconds] (static) 2.995 - 0.971 (moving)
FILE_3_STATIC_CAMERA_DELAY = 2.275  # [seconds] (static) 3.355 - 1.08 (moving)
FILE_4_STATIC_CAMERA_DELAY = 2.015  # [seconds] (static) 2.960 - 0.945 (moving)

# Setup based on choosen coin
if CHOOSEN_COIN == 1:
    CHOOSEN_VIDEO_STATIC_PATH = COIN_1_VIDEO_CAMERA_STATIC_PATH
    CHOOSEN_VIDEO_MOVING_PATH = COIN_1_VIDEO_CAMERA_MOVING_PATH
    CHOOSEN_VIDEO_DELAY = FILE_1_STATIC_CAMERA_DELAY
elif CHOOSEN_COIN == 2:
    CHOOSEN_VIDEO_STATIC_PATH = COIN_2_VIDEO_CAMERA_STATIC_PATH
    CHOOSEN_VIDEO_MOVING_PATH = COIN_2_VIDEO_CAMERA_MOVING_PATH
    CHOOSEN_VIDEO_DELAY = FILE_2_STATIC_CAMERA_DELAY
elif CHOOSEN_COIN == 3:
    CHOOSEN_VIDEO_STATIC_PATH = COIN_3_VIDEO_CAMERA_STATIC_PATH
    CHOOSEN_VIDEO_MOVING_PATH = COIN_3_VIDEO_CAMERA_MOVING_PATH
    CHOOSEN_VIDEO_DELAY = FILE_3_STATIC_CAMERA_DELAY
else:
    CHOOSEN_VIDEO_STATIC_PATH = COIN_4_VIDEO_CAMERA_STATIC_PATH
    CHOOSEN_VIDEO_MOVING_PATH = COIN_4_VIDEO_CAMERA_MOVING_PATH
    CHOOSEN_VIDEO_DELAY = FILE_4_STATIC_CAMERA_DELAY


def main():
    print("Analysis")

    static_video = cv.VideoCapture(CHOOSEN_VIDEO_STATIC_PATH)
    moving_video = cv.VideoCapture(CHOOSEN_VIDEO_MOVING_PATH)

    FPS_DIFFERENCE = MOVING_VIDEO_FPS - STATIC_VIDEO_FPS

    # Syncing the static video to the moving video by skipping the
    static_video.set(cv.CAP_PROP_POS_FRAMES, int(
        STATIC_VIDEO_FPS * CHOOSEN_VIDEO_DELAY))


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
