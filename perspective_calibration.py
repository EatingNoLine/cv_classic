import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

savedir = "cam_data/"
cam_mtx = np.load(savedir + "cam_mtx.npy")
dist = np.load(savedir + "dist.npy")
newcam_mtx = np.load(savedir + "newcam_mtx.npy")
inv_newcam_mtx = np.load(savedir + "inv_newcam_mtx.npy")
roi = np.load(savedir + "roi.npy")

wldpoints = None
imgpoints = None

wldpoints = np.array([[60, 60, -99.08], [60, 50, -99.08], [60, 40, -99.08], [60, 30, -99.08],
                      [60, 20, -99.08], [60, 10, -99.08], [60, 0, -99.08], [50, 60, -99.08],
                      [50, 50, -99.08], [50, 40, -99.08], [50, 30, -99.08], [50, 20, -99.08],
                      [50, 10, -99.08], [50, 0, -99.08]], dtype = np.float32)


img1 = cv2.imread("img1.jpg")
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray_img1, (7, 7), None)
if ret:
    corners2 = cv2.cornerSubPix(gray_img1, corners, (11, 11), (-1, -1), criteria)
    imgpoints = corners2[ :14]
    print(imgpoints)
    cv2.drawChessboardCorners(img1, (7, 7), corners2, ret)
    cv2.imshow("test" ,img1)
    cv2.waitKey(-1)

# solvePnP
ret, rvec1, tvec1, = cv2.solvePnP(wldpoints, imgpoints, newcam_mtx, dist)

print("rvec1")
print(rvec1)
np.save(savedir + "rvec1.npy", rvec1)

print("tvec1")
print(tvec1)
np.save(savedir + "tvec1.npy", tvec1)

R_mtx , jac = cv2.Rodrigues(rvec1)
print("R_mtx")
print(R_mtx)
np.save(savedir + "R_mtx.npy", R_mtx)

inv_R_mtx = np.linalg.inv(R_mtx)
print("inv_R_mtx")
print(inv_R_mtx)
np.save(savedir + "inv_R_mtx.npy", inv_R_mtx)

Rt = np.column_stack((R_mtx, tvec1))
print("Rt")
print(Rt)
np.save(savedir + "Rt.npy", Rt)

P_mtx = newcam_mtx.dot(Rt)
print("P_mtx")
print(P_mtx)
np.save(savedir + "P_mtx", P_mtx)

s_arr = np.array([0], dtype = np.float32)

for i in range(0, 14):
    XYZ1 = np.array([[wldpoints[i, 0], wldpoints[i, 1], wldpoints[i, 2], 1]], dtype = np.float32)
    XYZ1 = XYZ1.T
    suv1 = P_mtx.dot(XYZ1)
    s = suv1[2, 0]
    print(s)
    s_arr = np.array([s / 14 + s_arr[0]], dtype = np.float32)

# test

testimgpoint = corners2[48]
uv_1 = np.array([[testimgpoint[0, 0], testimgpoint[0, 1], 1]], dtype = np.float32)
uv_1 = uv_1.T
suv1 = s_arr[0] * uv_1
xyz_c = inv_newcam_mtx.dot(suv1)
xyz_c = xyz_c - tvec1
XYZ = inv_R_mtx.dot(xyz_c)
print("xyz")
s2 = 101.08 / XYZ[2, 0]
XYZ = s2 * XYZ
print(XYZ)
