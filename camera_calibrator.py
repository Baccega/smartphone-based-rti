import numpy as np
import cv2 as cv

print("Camera calibration")

VIDEO_CAMERA_CHOOSEN = 2

CHESSBOARD_SIZE = (9, 6)
SKIP_INTERVAL = 20

VIDEO_CAMERA_STATIC_PATH = 'data/cam1-static/calibration.mov'
VIDEO_CAMERA_MOVING_PATH = 'data/cam2-moving_light/calibration.mp4'

if VIDEO_CAMERA_CHOOSEN == 1:
    CHOOSEN_VIDEO = VIDEO_CAMERA_STATIC_PATH
else:
    CHOOSEN_VIDEO = VIDEO_CAMERA_MOVING_PATH


def main():
    calibration_video = cv.VideoCapture(CHOOSEN_VIDEO)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                        0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    frame_count = 0
    success = 1
    video_length = int(calibration_video.get(cv.CAP_PROP_FRAME_COUNT))
    while success:
        frame_count += 1
        success, frame = calibration_video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret)
            cv.imshow('img', frame)
            cv.waitKey(1)
        if(SKIP_INTERVAL * frame_count >= video_length):
            break
        calibration_video.set(cv.CAP_PROP_POS_FRAMES, SKIP_INTERVAL * frame_count)

    cv.destroyAllWindows()
    calibration_video.release()


    # images = glob.glob('data/*.png')
    # for fname in images:
    #     img = cv.imread(fname)

    # cv.destroyAllWindows()


    ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)


    # Displaying required output
    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)


if __name__ == "__main__":
    main()