import numpy as np
import cv2
import glob
import sys

print(cv2.__version__)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

grid_size = (5,9)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('G:/mldata/camera-calibration/lowpaper/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    #gray = cv2.resize(gray, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    #cv2.imwrite("xx.png", gray)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(gray, grid_size, corners, ret)
        cv2.imshow('img', gray)
        cv2.waitKey(10)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("mtx: ", mtx)
print("dist: ", dist)

img = cv2.imread(images[0])
h,  w = img.shape[:2]
#h = h // 4
#w = w // 4
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)