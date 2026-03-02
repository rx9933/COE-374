import cv2
import numpy as np
import glob

CHECKERBOARD = (9,6)
square_size = 0.025  # meters

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob('calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Reprojection error:", ret)
print("K:\n", K)
print("Dist:\n", dist)

np.save("K.npy", K)
np.save("dist.npy", dist)
