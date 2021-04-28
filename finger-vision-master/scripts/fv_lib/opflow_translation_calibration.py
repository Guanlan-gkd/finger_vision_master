#! /usr/bin/env python


import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from undistort_crop_resize import *
import flow_vis
from pynhhd.nHHD import nHHD
from copy import deepcopy


def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=0.0):
    # Don't affect original image
    image = image.copy()

    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, optical_flow_image.shape[1], 5), range(0, optical_flow_image.shape[0], 5)), 2)
    flow_end = (optical_flow_image[flow_start[:, :, 1],
                                   flow_start[:, :, 0], :] + flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(63, 208, 244),
                        thickness=1,
                        tipLength=.1)
    return image


if __name__ == "__main__":

    calibration_folder_name = "/home/yipai/data_11_27/4x4/times/*.jpg"
    # calibration_folder_name = "/home/yipai/image_data/translation_calibration/time1/*.jpg"
    print("calibration folder: " + calibration_folder_name)
    calibration_file_paths = sorted(glob.glob(calibration_folder_name))
    calibration_file_paths = calibration_file_paths[1:]
    calibration_step = 0.05

    # test_folder_name = "/home/yipai/image_data/translation_calibration/time1/*.jpg"
    test_folder_name = "/home/yipai/data_11_13/dis_1_time/*.jpg"
    # test_folder_name = "/home/yipai/data_11_27/3x3/times/*.jpg"
    print("test folder: " + test_folder_name)
    test_file_paths = sorted(glob.glob(test_folder_name))
    test_file_paths = test_file_paths[1:]
    test_step = 0.1

    # DIS method
    dis_inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # dis_inst.setFinestScale(2)  # 2 as default
    # dis_inst.setPatchStride(3)  # 4 as default
    dis_inst.setGradientDescentIterations(25)  # 12 as default
    dis_inst.setVariationalRefinementIterations(10)  # 0 as default
    # dis_inst.setVariationalRefinementAlpha(0.0)
    # dis_inst.setPatchSize(15)  # 8 as default

    # # decomp object
    # dx = float(1.0)
    # dy = float(1.0)
    # grid = (360, 480)
    # # grid = (480, 640)
    # decomposer = nHHD(grid=grid, spacings=(dy, dx))

    flow = None
    calibration_mag_points = []
    for i in range(len(calibration_file_paths)):
        img = cv2.imread(calibration_file_paths[i], cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        if i == 0:
            base = frame
        two_frame_flow = dis_inst.calc(frame, base, flow)
        flow = two_frame_flow
        cropped_flow = flow[60:-60, 80:-80]
        flow_on_image = np.hypot(cropped_flow[:, :, 0], cropped_flow[:, :, 1])
        flow_on_image = flow_on_image.flatten()
        sorted_disp_mag = np.sort(flow_on_image)
        mag = sorted_disp_mag[-10:].mean()
        calibration_mag_points.append(mag)

    calibration_mag_points = np.array(calibration_mag_points)
    num_of_points = len(calibration_mag_points)
    depth_points = np.linspace(
        0, (num_of_points-1)*calibration_step, num_of_points)
    p = np.polyfit(calibration_mag_points, depth_points, 1).reshape(1, 2)
    print("p[0]: ", p[0, 0], " p[1]: ", p[0, 1])

    flow = None
    test_mag_points = []
    for i in range(len(test_file_paths)):
        img = cv2.imread(test_file_paths[i], cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        if i == 0:
            base = frame
        two_frame_flow = dis_inst.calc(frame, base, flow)
        flow = two_frame_flow
        cropped_flow = flow[60:-60, 80:-80]
        flow_on_image = np.hypot(cropped_flow[:, :, 0], cropped_flow[:, :, 1])
        flow_on_image = flow_on_image.flatten()
        indicators = np.where(cropped_flow[:, :, 0].flatten() > 0.0)
        flow_on_image[indicators] = 0.0
        sorted_disp_mag = np.sort(flow_on_image)
        mag = sorted_disp_mag[-10:].mean()
        test_mag_points.append(mag)

    test_mag_points = np.array(test_mag_points)
    num_of_points = len(test_mag_points)
    t = np.concatenate((test_mag_points.reshape(
        1, -1), np.ones((1, num_of_points))), axis=0)
    predictions = (np.matmul(p, t)).flatten()
    ground_truth = np.linspace(0, (num_of_points-1)*test_step, num_of_points)
    plt.figure()
    plt.plot(test_mag_points, predictions, label="prediction")
    plt.plot(test_mag_points, ground_truth, label="ground truth")
    plt.legend()
    coeff = np.corrcoef(np.array([test_mag_points, ground_truth]))
    plt.title("testing data correlation coefficient: " + str(coeff[0, 1]))
    plt.show()
    print("testing correlation coefficient: ", coeff[0, 1])
