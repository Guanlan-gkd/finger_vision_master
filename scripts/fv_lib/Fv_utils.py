import glob
import os
import time
from copy import deepcopy

import cv2
import freud
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import mode, multivariate_normal
# from gf import guided_filter


def streamFromImages(folder_name):
    file_names = sorted(glob.glob(folder_name))
    imgs = []
    for filename in file_names:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return imgs


def getImageNames(folder_name):
    file_names = sorted(glob.glob(folder_name))
    base_names = []
    for f in file_names:
        base_names.append(os.path.basename(f))
    return base_names


def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=0.0):
    # Don't affect original image
    image = image.copy()
    optical_flow_image = (optical_flow_image).copy()
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image, ) * 3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(range(0, optical_flow_image.shape[1], 10),
                    range(0, optical_flow_image.shape[0], 10)), 2)
    flow_end = (
        optical_flow_image[flow_start[:, :, 1], flow_start[:, :, 0], :] +
        flow_start).astype(np.int32)

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
                        color=(255, 255, 204),
                        thickness=1,
                        tipLength=.1)
    return image


threshold = None


def correctInvalidArea(density):
    # density = guided_filter(density, density, 5, 0.001, s=4)
    # invalid_ind = np.zeros_like(density, dtype='uint8')
    global threshold
    if threshold is None:
        threshold = np.median(density)
    invalid_ind = twoClustering(density, threshold)
    density = cv2.inpaint(density * 1e4, invalid_ind, 5,
                          cv2.INPAINT_TELEA) / 1e4
    # density -= threshold
    # density[np.where(density<0)]=0
    # density = guided_filter(density, density, 5, 0.003, s=4)
    return density
    # return invalid_ind


def twoClustering(density, threshold):
    minThreshold = threshold
    cluster = np.zeros_like(density, dtype='uint8')
    cluster[np.where(density < minThreshold)] = 1
    kernel = np.ones((5, 5), np.uint8)
    cluster = cv2.dilate(cluster, kernel, iterations=2)
    return cluster


# def getForwardBackwardError(I0, I1, flow):
#     h, w = I0.shape[:2]
#     # h /= 10
#     # w /= 10
#     # I0 = cv2.resize(I0, (w, h), interpolation=cv2.INTER_AREA).copy()
#     # I1 = cv2.resize(I1, (w, h), interpolation=cv2.INTER_AREA).copy()
#     # flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_AREA).copy()
#     x = np.linspace(0, w-1, w)
#     y = np.linspace(0, h-1, h)
#     interploator = RegularGridInterpolator((x, y), I1.T.astype(
#         'float32'), method='linear', bounds_error=False, fill_value=mode(I0, axis=None)[0])

#     points = np.array(np.meshgrid(x, y, indexing='ij')).transpose(2, 1, 0)
#     points += flow
#     # points = points[20:-20, 20:-20]
#     # I0 = I0[20:-20, 20:-20].copy()
#     points = points.reshape((-1, 2))

#     z = interploator(points)
#     z[np.where(z > 255.0)] = 255.0
#     z[np.where(z < 0.0)] = 0.0
#     z = z.reshape(I0.shape)

#     rev = np.abs((z-I0.astype("float32")))  # [15:-15, 20:-20].copy()
#     # h *= 2
#     # w *= 2
#     # rev = cv2.resize(rev, (w, h), interpolation=cv2.INTER_AREA)
#     # rev[np.where(rev < 30)] = 0
#     rev = rev.astype('uint8')
#     invalid_ind = np.zeros_like(rev)
#     invalid_ind[np.where(rev > 20.0)] = 1
#     ind = distance_transform_edt(
#         invalid_ind, return_distances=False, return_indices=True)
#     flow_x = flow[:, :, 0]
#     flow_y = flow[:, :, 1]
#     flow_x = flow_x[tuple(ind)]
#     flow_y = flow_y[tuple(ind)]
#     corrected_flow = np.stack((flow_x, flow_y), axis=2)
#     # rev = cv2.resize(rev, (w*10, h*10))
#     # rev = cv2.medianBlur(rev, 3)
#     # rev = cv2.applyColorMap(rev, cv2.COLORMAP_HOT)

#     # moments = cv2.moments(rev)
#     # x_bar = moments['m10'] / moments['m00']
#     # y_bar = moments['m01'] / moments['m00']
#     # cov = np.array([[moments['mu20'], moments['mu11']], [
#     #                    moments['mu11'], moments['mu02']]]) / moments['m00']
#     # X, Y = np.meshgrid(x, y)
#     # coordinates = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
#     # pdf = multivariate_normal.pdf(coordinates, mean=[x_bar, y_bar], cov=cov)
#     # pdf = ((pdf / 1e-5 * 255)).astype("uint8").reshape(h, w)
#     # pdf = cv2.applyColorMap(pdf, cv2.COLORMAP_HOT)
#     # print(x_bar, y_bar)
#     # print(cov)

#     # plt.clf()
#     # plt.imshow(pdf)
#     # print(x_bar, y_bar, sigma_2)
#     # print(rev.max())
#     # print(rev.min())
#     # plt.clf()
#     # plt.hist(pdf.flatten(), bins=256)
#     # rev = np.hypot(flow[:, :, 0], flow[:, :, 1])/50*255
#     # rev = cv2.applyColorMap(rev.astype("uint8"), cv2.COLORMAP_HOT)
#     # plt.clf()
#     # # plt.hist(rev.flatten(), bins=256)
#     # plt.imshow(rev)

#     # z = cv2.resize(z, (w, h), interpolation=cv2.INTER_AREA)
#     # I1 = cv2.resize(I1, (w, h), interpolation=cv2.INTER_AREA)

#     # cv2.imshow("interp", z)
#     # cv2.imshow("base", I0)
#     # cv2.imshow("original", I1)

#     # plt.savefig("/home/yipai/flows/" + str(time.time()) + ".jpg")
#     return rev, corrected_flow


class Tracker:
    def __init__(self, adaptive=False, isFast=True):
        if isFast:
            # DIS method
            self.__tracking_inst = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            # self.__tracking_inst.setFinestScale(1)  # 2 as default
            # self.__tracking_inst.setPatchStride(4)  # 4 as default
            # self.__tracking_inst.setGradientDescentIterations(20)  # 12 as default
            # self.__tracking_inst.setVariationalRefinementIterations(4)  # 0 as default
            # self.__tracking_inst.setVariationalRefinementAlpha(0.0)
            # self.__tracking_inst.setPatchSize(15)  # 8 as default

            self.__coarse_tracking_inst = cv2.DISOpticalFlow_create(
                cv2.DISOpticalFlow_PRESET_ULTRAFAST)
        else:
            # #deep flow method
            self.__tracking_inst = cv2.optflow.createOptFlow_DeepFlow()
            self.__coarse_tracking_inst = self.__tracking_inst
        self.__fine_base_frame_queue = []
        self.__coarse_base_frame_queue = []
        self.__fine_flow_queue = []
        self.__coarse_flow_queue = []
        self.__prev_fine_flow = None
        self.__prev_coarse_flow = None
        self.__adaptive = adaptive

    def track(self, frame):
        if self.__adaptive:
            if not self.__fine_base_frame_queue:
                self.__fine_base_frame_queue.append(frame)
                self.__coarse_base_frame_queue.append(
                    cv2.resize(frame, (20, 15), interpolation=cv2.INTER_AREA))
                self.__fill_value = mode(self.__coarse_base_frame_queue[-1],
                                         axis=None)[0]
                self.__fine_flow_queue.append(
                    np.zeros((frame.shape[0], frame.shape[1], 2)))
                self.__coarse_flow_queue.append(np.zeros((15, 20, 2)))
                self.__prev_fine_flow = np.zeros(
                    (frame.shape[0], frame.shape[1], 2))
                self.__prev_coarse_flow = np.zeros((15, 20, 2))
                return self.__fine_flow_queue[-1].copy()
            else:
                downScaled_frame = cv2.resize(frame, (20, 15),
                                              interpolation=cv2.INTER_AREA)
                self.__prev_coarse_flow = self.__coarse_tracking_inst.calc(
                    self.__coarse_base_frame_queue[-1], downScaled_frame,
                    self.__prev_coarse_flow)

                # if self.__checkLoopClosure(downScaled_frame) > 0:
                #     pass
                self.__prev_fine_flow = self.__tracking_inst.calc(
                    self.__fine_base_frame_queue[-1], frame,
                    self.__prev_fine_flow)
                error = self.__getForwardBackwardError(
                    self.__coarse_base_frame_queue[-1], downScaled_frame,
                    self.__prev_coarse_flow)
                if error[2:-2, 2:-2].max() > 10:
                    print("new base frame inserted")
                    self.__fine_base_frame_queue.append(frame)
                    self.__coarse_base_frame_queue.append(downScaled_frame)
                    self.__fill_value = mode(
                        self.__coarse_base_frame_queue[-1], axis=None)[0]
                    self.__fine_flow_queue.append(self.__fine_flow_queue[-1] +
                                                  self.__prev_fine_flow)
                    self.__prev_fine_flow = None
                    self.__coarse_flow_queue.append(
                        self.__coarse_flow_queue[-1] + self.__prev_coarse_flow)
                    self.__prev_coarse_flow = None
                    return self.__fine_flow_queue[-1].copy()
                return (self.__fine_flow_queue[-1] +
                        self.__prev_fine_flow).copy()
        else:
            if not self.__fine_base_frame_queue:
                self.__fine_base_frame_queue.append(frame)
            self.__prev_fine_flow = self.__tracking_inst.calc(
                self.__fine_base_frame_queue[-1], frame, self.__prev_fine_flow)
            return self.__prev_fine_flow

    def __getForwardBackwardError(self, I0, I1, flow):
        h, w = I0.shape[:2]
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        interploator = RegularGridInterpolator((x, y),
                                               I1.T.astype('float'),
                                               method='nearest',
                                               bounds_error=False,
                                               fill_value=self.__fill_value)

        points = np.array(np.meshgrid(x, y, indexing='ij')).transpose(2, 1, 0)
        points += flow
        points = points.reshape((-1, 2))

        z = interploator(points)
        z[np.where(z > 255.0)] = 255.0
        z[np.where(z < 0.0)] = 0.0
        z = z.reshape(I0.shape)

        rev = np.abs((z - I0.astype("float")))

        return rev

    def __checkLoopClosure(
        self, downScaled_frame
    ):  # return id of the closest frame forming the loop, -1 if no loop found
        total_flow = self.__prev_coarse_flow + self.__coarse_flow_queue[-1]
        for i in range(len(self.__coarse_base_frame_queue) - 1):
            error = self.__getForwardBackwardError(
                self.__coarse_base_frame_queue[i], downScaled_frame,
                total_flow - self.__coarse_flow_queue[i])
            if error[2:-2, 2:-2].max() < 10:
                print("loop closed at frame ", str(i))
                self.__fine_base_frame_queue = self.__fine_base_frame_queue[:
                                                                            i +
                                                                            1]
                self.__coarse_base_frame_queue = self.__coarse_base_frame_queue[:
                                                                                i
                                                                                +
                                                                                1]
                self.__fill_value = mode(self.__coarse_base_frame_queue[-1],
                                         axis=None)[0]
                self.__prev_fine_flow = self.__prev_fine_flow + self.__fine_flow_queue[
                    -1] - self.__fine_flow_queue[i]
                self.__fine_flow_queue = self.__fine_flow_queue[:i + 1]
                self.__prev_coarse_flow = self.__prev_coarse_flow + self.__coarse_flow_queue[
                    -1] - self.__coarse_flow_queue[i]
                self.__coarse_flow_queue = self.__coarse_flow_queue[:i + 1]

                return i
        return -1

    def reset(self):
        self.__fine_base_frame_queue = []
        self.__coarse_base_frame_queue = []
        self.__fine_flow_queue = []
        self.__coarse_flow_queue = []
        self.__prev_fine_flow = None
        self.__prev_coarse_flow = None


def getDensity(flow):
    sigma = 3.0
    r_max = 9.0

    x = np.linspace(-30 - (flow.shape[1] - 1) / 2,
                    30 + (flow.shape[1] - 1) / 2,
                    flow.shape[1] + 60,
                    endpoint=True,
                    dtype='float32')
    y = np.linspace(-30 - (flow.shape[0] - 1) / 2,
                    30 + (flow.shape[0] - 1) / 2,
                    flow.shape[0] + 60,
                    endpoint=True,
                    dtype='float32')
    coordinate = np.array(np.meshgrid(x, y), dtype='float32').transpose(
        (1, 2, 0))
    padded_flow = np.zeros((flow.shape[0] + 60, flow.shape[1] + 60, 2),
                           dtype="float32")
    padded_flow[30:-30, 30:-30, :] = flow
    points = (coordinate + padded_flow).reshape(-1, 2)
    points = np.hstack((points, np.zeros((points.shape[0], 1))))

    box = freud.box.Box(flow.shape[1] - 1 + 30, flow.shape[0] - 1 + 30)
    box.periodic = np.array((False, False, False))

    gd = freud.density.GaussianDensity(
        (flow.shape[1] + 60, flow.shape[0] + 60), r_max, sigma)
    system = freud.LinkCell(box, points, cell_width=r_max)
    # system = freud.AABBQuery(box, points)
    gd.compute(system)

    # density = deepcopy(-gd.density.T)
    density = deepcopy((-gd.density.T)[30:-30, 30:-30])
    density = correctInvalidArea(density)
    return density