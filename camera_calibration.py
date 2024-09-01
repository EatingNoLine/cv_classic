import numpy as np
import cv2
import glob
import time

workingdir = "/USTC/RoboGame/cv"
savedir = "cam_data/"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7 * 7, 3), np.float32)
objp[ :, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)*10

objpoints = []
imgpoints = []
images = glob.glob("calibration_images/*.jpg")

win_name="Verify"
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

for fname in images:
    img = cv2.imread(fname)
    print(fname)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_img, (7, 7), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
        cv2.imshow(win_name, img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

img1 = cv2.imread("img1.jpg")

print("start calibration")
ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[ : :-1], None, None)

print("cam_mtx")
print(cam_mtx)
np.save(savedir + "cam_mtx.npy", cam_mtx)

print("dist")
print(dist)
np.save(savedir + "dist.npy", dist)


h, w = img1.shape[ :2]
print(w, h)
newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))

print("roi")
print(roi)
np.save(savedir + "roi.npy", roi)

print("newcam_mtx")
print(newcam_mtx)
np.save(savedir + "newcam_mtx.npy", newcam_mtx)

inv_newcam_mtx = np.linalg.inv(newcam_mtx)
print("inv_newcam_mtx")
print(inv_newcam_mtx)
np.save(savedir + "inv_newcam_mtx", inv_newcam_mtx)

undst = cv2.undistort(img1, cam_mtx, dist, None, newcam_mtx)
x, y, w, h = roi
undst = undst[y : y + h, x : x + w]

cv2.imshow("img1", img1)
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imshow("undst", undst)
cv2.waitKey(5000)
cv2.destroyAllWindows()

