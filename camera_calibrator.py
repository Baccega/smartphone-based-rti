import numpy as np
import cv2 as cv
import constants

def calibrate(calibration_video, save_file_path):
    calibration_video = cv.VideoCapture(calibration_video)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(
        (constants.CHESSBOARD_SIZE[0] * constants.CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:constants.CHESSBOARD_SIZE[0],
                           0:constants.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
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
        ret, corners = cv.findChessboardCorners(
            gray, constants.CHESSBOARD_SIZE, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(
                frame, constants.CHESSBOARD_SIZE, corners2, ret)
            cv.imshow('img', frame)
            cv.waitKey(1)
        if(constants.CALIBRATION_FRAME_SKIP_INTERVAL * frame_count >= video_length):
            break
        calibration_video.set(
            cv.CAP_PROP_POS_FRAMES, constants.CALIBRATION_FRAME_SKIP_INTERVAL * frame_count)

    cv.destroyAllWindows()
    calibration_video.release()

    # images = glob.glob('data/*.png')
    # for fname in images:
    #     img = cv.imread(fname)

    # cv.destroyAllWindows()

    ret, K, dist, r_vecs, t_vecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # Displaying required output
    print("Camera matrix:")
    print(K)

    print("\n Distortion coefficient:")
    print(dist)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)

    Kfile = cv.FileStorage(save_file_path, cv.FILE_STORAGE_WRITE)
    Kfile.write("RMS", ret)
    Kfile.write("K", K)
    Kfile.write("dist", dist)
    Kfile.release()
    print("Saved intrinsics!")


def main():
    print("Camera calibration")
    calibrate(constants.CALIBRATION_CAMERA_STATIC_PATH,
              constants.CALIBRATION_INTRINSICS_CAMERA_STATIC_PATH)
    calibrate(constants.CALIBRATION_CAMERA_MOVING_PATH,
              constants.CALIBRATION_INTRINSICS_CAMERA_MOVING_PATH)


if __name__ == "__main__":
    main()
