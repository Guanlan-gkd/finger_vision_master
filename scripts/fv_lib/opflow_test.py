#! /usr/bin/env python


import glob
import os
import time

import cv2
import freud
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
from scipy.stats import entropy
from scipy.io import savemat
# from undistort_crop_resize import *
import flow_vis
from pynhhd.nHHD import nHHD
from copy import deepcopy


def streamFromImages(folder_name):
    file_names = sorted(glob.glob(folder_name))
    imgs = []
    for filename in file_names:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return imgs


def streamFromVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    imgs = []
    success, img = cap.read()
    while success is True:
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        success, img = cap.read()
    cap.release()
    return imgs


def getImageNames(folder_name):
    file_names = sorted(glob.glob(folder_name))
    base_names = []
    for f in file_names:
        base_names.append(os.path.basename(f))
    return base_names


# def getTranslationFlowError(flows):
#     n = len(flows)
#     cropped_flows = []
#     for flow in flows:
#         # crop the original flow to removed lost points
#         cropped_flows.append(flow[100:-100, 100:-100])
#     cropped_flows = np.concatenate(cropped_flows).reshape((n, -1, 2))
#     mean_motion = cropped_flows.mean(axis=1, keepdims=True)
#     error = np.sqrt(np.square((cropped_flows - mean_motion)
#                               ).sum(axis=2)).mean(axis=1)
#     return error


# def getFlowPolarEntropy(flow):
#     bins = 100
#     m = 40.0
#     flow = flow.copy()[100:-100, 100:-100]
#     mag = np.sqrt(np.square(flow.reshape((-1, 2))).sum(axis=1))
#     phase = np.arctan2(flow[:, :, 0], flow[:, :, 1]).flatten()
#     phase_hist, _ = np.histogram(
#         phase, bins=180, range=(-np.pi, np.pi), density=True)
#     phase_hist /= 180/2/np.pi
#     mag_max = max(mag)
#     upper_bound = max(mag_max, m)
#     mag_hist, _ = np.histogram(
#         mag, bins=bins, range=(0.0, upper_bound), density=True)
#     mag_hist /= bins/upper_bound
#     return entropy(mag_hist), entropy(phase_hist)


# def getFlowCartEntropy(flow, bins=100, m=40.0):
#     bins = 100
#     flow = flow.copy()[100:-100, 100:-100]
#     x = flow[:, :, 0].flatten()
#     y = flow[:, :, 0].flatten()
#     m = max(np.abs(x).max(), np.abs(y).max(), 40.0)
#     x_hist, _ = np.histogram(x, bins=bins, range=(-m, m), density=True)
#     x_hist /= bins/2/m
#     y_hist, _ = np.histogram(y, bins=bins, range=(-m, m), density=True)
#     y_hist /= bins/2/m
#     return entropy(x_hist), entropy(y_hist)


def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=0.0):
    # Don't affect original image
    image = image.copy()
    optical_flow_image = (optical_flow_image*5).copy()
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, optical_flow_image.shape[1], 40), range(0, optical_flow_image.shape[0], 40)), 2)
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
                        color=(255, 255, 204),
                        thickness=2,
                        tipLength=.3)
    return image


# def findFlowCenter(flows):
#     w = flows[0].shape[1]
#     cropped_flows = []
#     for flow in flows:
#         # crop the original flow to removed lost points
#         cropped_flows.append(flow[100:-100, 100:-100])
#     w = cropped_flows[0].shape[1]
#     centers = []
#     weights = []
#     for flow in cropped_flows:
#         mag = np.sqrt(np.square(flow).sum(axis=2))
#         c = np.argmin(mag)
#         centers.append(divmod(c, w))
#         weights.append(mag.mean())
#     centers = np.array(centers)
#     centers += 100
#     weights = np.power(1.1, np.array(weights))-1
#     weights /= weights.sum()
#     centers = np.multiply(centers, weights.reshape(-1, 1)).sum(axis=0)
#     return centers  # return the center of rotation [y_c, x_c]


# def getRotationError(flows, rotate_step=1.0):
#     # @params rotate_step the rotated degree at each step, + for clockwise
#     rotation_center = findFlowCenter(flows)
#     crop_window = [100, -100, 100, -100]
#     errors = []
#     rotate_radian = np.radians(rotate_step)
#     x = np.linspace(0, flows[0].shape[1]-1, flows[0].shape[1])
#     y = np.linspace(0, flows[0].shape[0]-1, flows[0].shape[0])
#     coordinate = np.array(np.meshgrid(x, y)).transpose((1, 2, 0))
#     coordinate[:, :, 0] -= rotation_center[1]
#     coordinate[:, :, 1] -= rotation_center[0]
#     r = np.sqrt(np.square(coordinate).sum(axis=2))
#     init_phi = np.arctan2(coordinate[:, :, 1], coordinate[:, :, 0])
#     for i, flow in enumerate(flows):
#         phi = i*rotate_radian + init_phi
#         x_hat = np.multiply(r, np.cos(phi))
#         y_hat = np.multiply(r, np.sin(phi))
#         flow_hat = np.transpose(np.array([x_hat, y_hat]), [
#                                 1, 2, 0]) - coordinate
#         error = np.sqrt(np.square(flow_hat-flow).sum(axis=2))
#         error = error[crop_window[0]:crop_window[1],
#                       crop_window[2]:crop_window[3]]
#         errors.append(error.mean())
#     return errors


# def getRotationCorrelations(flows):
#     center = findFlowCenter(flows)
#     Rs = []
#     crop_window = [100, -100, 100, -100]
#     x = np.linspace(0, flows[0].shape[1]-1, flows[0].shape[1])
#     y = np.linspace(0, flows[0].shape[0]-1, flows[0].shape[0])
#     coordinate = np.array(np.meshgrid(x, y)).transpose((1, 2, 0))
#     coordinate[:, :, 0] -= center[1]
#     coordinate[:, :, 1] -= center[0]
#     r = np.sqrt(np.square(coordinate).sum(axis=2))
#     r = r[crop_window[0]:crop_window[1],
#           crop_window[2]:crop_window[3]].flatten()
#     phi = np.arctan2(coordinate[:, :, 1], coordinate[:, :, 0])
#     phi = phi[crop_window[0]:crop_window[1],
#               crop_window[2]:crop_window[3]].flatten()
#     sort_ind = np.argsort(phi)
#     phi = phi[sort_ind]
#     for flow in flows[1:]:
#         cropped_flow = flow[crop_window[0]:crop_window[1],
#                             crop_window[2]:crop_window[3]]
#         flow_mag = np.sqrt(np.square(cropped_flow).sum(axis=2)).flatten()
#         mag_R = np.corrcoef(np.array([r, flow_mag]))
#         flow_angle = np.arctan2(
#             cropped_flow[:, :, 1], cropped_flow[:, :, 0]).flatten()
#         flow_angle = flow_angle[sort_ind]
#         intercept = flow_angle[:100].min()
#         flow_angle[np.where(flow_angle < intercept)] += 2*np.pi
#         angle_R = np.corrcoef(np.array([phi, flow_angle]))
#         # plt.plot(phi, flow_angle, "go")
#         # plt.show(block=True)
#         Rs.append([mag_R[0, 1], angle_R[0, 1]])
#     return np.array(Rs)


def getVoronoiAreas(flow):
    original_shape = flow.shape
    flow = cv2.resize(
        flow, (flow.shape[0] // 10, flow.shape[1] // 10), interpolation=cv2.INTER_AREA)
    x = np.linspace(-(original_shape[1]-1) / 2,
                    (original_shape[1]-1) / 2, flow.shape[1])
    y = np.linspace(-(original_shape[0]-1) / 2,
                    (original_shape[0]-1) / 2, flow.shape[0])
    coordinate = np.array(np.meshgrid(x, y)).transpose((1, 2, 0))
    points = (coordinate + flow).reshape(-1, 2)
    points = np.hstack((points, np.zeros((points.shape[0], 1))))

    # # python3 test
    # start_time = time.time()
    # box = freud.box.Box.square(2*original_shape[1]/10)
    # voro = freud.locality.Voronoi()
    # voro.compute((box, points/10))
    # print("time used: ", str(time.time()-start_time))

    # python2 test
    start_time = time.time()
    box = freud.box.Box.square(2*original_shape[1])
    voro = freud.voronoi.Voronoi(box, original_shape[1])
    voro.compute(box=box, positions=points)
    voro.computeVolumes()
    print("time used: ", str(time.time()-start_time))

    area = deepcopy(voro.volumes).reshape(
        (flow.shape[0], flow.shape[1]))
    area = area[3:-3, 3:-3]
    area[np.where(area > 8.0)] = 8.0
    area = (area/8.0*255.0).astype(dtype='uint8')
    # area = cv2.ximgproc.l0Smooth(area, None, 0.001)
    area = cv2.resize(
        area, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
    depth_image = cv2.applyColorMap(area, cv2.COLORMAP_HOT)
    cv2.imwrite("/home/yipai/depth_imgs/" +
                str(time.time()) + ".jpg", depth_image)
    return area


if __name__ == "__main__":

    # folder_name = "/home/yipai/data_11_27/3x3/times/*.jpg"
    folder_name = "/home/yipai/image_data/translation_calibration/time1/*.jpg"
    print("folder: " + folder_name)
    file_paths = sorted(glob.glob(folder_name))
    file_paths = file_paths[1:13]
    # img_names = getImageNames(folder_name)
    # tik_folder_name = "/home/yipai/image_data/translation_calibration/time2/*.jpg"
    # tik_img_names = getImageNames(tik_folder_name)
    # tik_img_names = tik_img_names[1:]
    # video_name = "/home/yipai/translation_videos/2.mp4"
    # video_name = "/home/yipai/zoom_videos/4.mp4"
    # print("video name: " + video_name)
    # imgs = streamFromVideo(video_name)
    # cv2.imshow("testwindows", imgs[-1][60:-60, 80:-80])
    # cv2.waitKey(100000000)
    # DIS method
    dis_inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # dis_inst.setFinestScale(2)  # 2 as default
    # dis_inst.setPatchStride(3)  # 4 as default
    dis_inst.setGradientDescentIterations(25)  # 12 as default
    dis_inst.setVariationalRefinementIterations(10)  # 0 as default
    # dis_inst.setVariationalRefinementAlpha(0.0)
    # dis_inst.setPatchSize(15)  # 8 as default

    # video writer
    # out = cv2.VideoWriter("/home/yipai/out.avi", cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(
    #     'M', 'J', 'P', 'G'), 5, (imgs[0].shape[1], imgs[0].shape[0]*3))

    # # decomp object
    # dx = float(1.0)
    # dy = float(1.0)
    # # grid = (360, 480)
    # grid = (480, 640)
    # decomposer = nHHD(grid=grid, spacings=(dy, dx))

    start_time = time.time()
    # base = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    # plt.figure("new")
    # plt.imshow(base, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    flow = None
    # for i in range(len(imgs)):
    #     frame = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
    #     flow = dis_inst.calc(base, frame, flow)
    # print("Pure Frequency of DIS: " +
    #       str(len(imgs)/(time.time()-start_time)) + " Hz")
    flow = None
    flows = []
    start_time = time.time()
    m = []
    areas = []
    area_points = []
    for i in range(len(file_paths)):
        # frame = cv2.cvtColor(cv2.resize(imgs[i], (int(imgs[i].shape[1]*1.1), int(
        #     imgs[i].shape[0]*1.1)), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
        # frame2 = cv2.resize(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY), (int(imgs[i].shape[1]*1.2), int(
        #     imgs[i].shape[0]*1.2)), interpolation=cv2.INTER_NEAREST)
        img = cv2.imread(file_paths[i])
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if i == 0:
            base = frame
            continue
        two_frame_flow = dis_inst.calc(base, frame, flow)
        # base = frame
        flow = two_frame_flow
        # flows.append(flow)
        # frame = np.concatenate((imgs[i], put_optical_flow_arrows_on_image(
        #     imgs[i], flow), flow_vis.flow_to_color(flow, convert_to_bgr=True), ), axis=0)
        # cropped_flow = flow[60:-60, 80:-80]
        # cropped_flow = deepcopy(flow)
        # cropped_flow[180-80:180+80, ]
        # decomposer.decompose(cropped_flow, verbose=0)
        # area_image = getVoronoiAreas(decomposer.d)
        # if img_names[i] == tik_img_names[0]:
            # if i % 10 == 0:
        # decomposer.decompose(flow, verbose=0)
        # area_image = getVoronoiAreas(cropped_flow)
        # areas.append(area_image)
        # flow_on_image = put_optical_flow_arrows_on_image(img, flow)
        # flow_on_image1 = put_optical_flow_arrows_on_image(img, decomposer.d)
        # flow_on_image2 = put_optical_flow_arrows_on_image(img, decomposer.r)
        # flow_on_image3 = put_optical_flow_arrows_on_image(img, decomposer.h)
        # edge = np.zeros((img.shape[0], img.shape[1]//10, 3), dtype="uint8")
        # flow_on_image = np.hstack((flow_on_image1, edge, flow_on_image2, edge, flow_on_image3))
    # # r_norm = np.sqrt(np.square(decomposer.r).sum(axis=2)).sum()
    # # d_norm = np.sqrt(np.square(decomposer.d).sum(axis=2)).sum()
    # # print(r_norm / d_norm)
        # cv2.imwrite("/home/yipai/flows/" +
        #         str(time.time()) + ".jpg", flow_on_image)
    # maxind = area_image.argmax()
    # r = maxind // area_image.shape[1]
    # c = maxind % area_image.shape[1]
    # print("max location: ", str(r), str(c))
    # area_points.append(area_image[r, c])
        # flow_on_image = put_optical_flow_arrows_on_image(imgs[i][60:-60, 80:-80], cropped_flow)
        # cv2.imwrite("/home/yipai/flows/" +
        #             str(time.time()) + ".jpg", flow_on_image)
        # flow_on_image = np.hypot(decomposer.d[:, :, 0], decomposer.d[:, :, 1])
        # flow_on_image = np.sort(flow_on_image.flatten())
        # flow_on_image = flow_on_image[-20:].mean()
        # area_points.append(flow_on_image)
        # flow_on_image = (flow_on_image/15*255).astype(dtype='uint8')
        # flow_on_image = cv2.applyColorMap(flow_on_image, cv2.COLORMAP_HOT)
    # # flow_on_image = cv2.resize(flow_on_image, (flow_on_image.shape[1], flow_on_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite("/home/yipai/flows/" +
    #             str(time.time()) + ".jpg", flow_on_image)
        # frame = np.concatenate((imgs[i], flow_on_image, area_image), axis=0)
        # out.write(frame)
        # if img_names[i] == base_name:
        #     base = frame
        #     flow = None
        #     continue
        # print(img_names[i], " ", tik_img_names[0], " ", str(img_names[i] == tik_img_names[0]))
        # if img_names[i] == tik_img_names[0]:
            # tik_img_names.pop(0)
        cropped_flow = flow[60:-60, 80:-80]
        # cv2.imwrite("/home/yipai/depth_imgs_times/" +
        #             str(time.time()) + ".jpg", area_image)
        # decomposer.decompose(cropped_flow, verbose=0)
        flow_on_image = np.hypot(cropped_flow[:, :, 0], cropped_flow[:, :, 1])
        flow_on_image = flow_on_image.flatten()
        sorted_disp_mag = np.sort(flow_on_image)
        flow_on_image = sorted_disp_mag[-100:].mean()
        area_points.append(flow_on_image)
            # if not tik_img_names:
            #     file_paths = file_paths[:i]
            #     break
    # out.release()
    a = getVoronoiAreas(cropped_flow)
    print("Frequency of all processing: " +
          str(len(file_paths)/(time.time()-start_time)) + " Hz")
    area_points = np.array(area_points)
    num_of_points = len(area_points)
    depth_points = np.linspace(
        0, (num_of_points-1)*0.1, num_of_points)
    p = np.polyfit(area_points, depth_points, 1).reshape(1,2)
    print("p[0]: ", p[0, 0], " p[1]: ", p[0, 1])
    t = np.concatenate((area_points.reshape(1, -1), np.ones((1, num_of_points))), axis=0)
    line_points = (np.matmul(p, t)).flatten()
    plt.figure()
    plt.plot(area_points, depth_points)
    plt.plot(area_points, line_points)
    coeff = np.corrcoef(np.array([area_points, depth_points]))
    plt.title("correlation coefficient: " + str(coeff[0, 1]))
    plt.show()
    print("correlation coefficient: ", coeff[0, 1])
    
    # # cv2.imwrite("/home/yipai/flows/" +
    # #                 str(time.time()) + ".jpg", imgs[-1][60:-60, 80:-80])
    # areas = np.array(areas).transpose((2, 0, 1))
    # savemat("/home/yipai/data.mat", mdict=(dict(areas=areas,
    #                                             depth_points=depth_points, area_points=area_points)))
    # m = np.array(m)
    # print("min: " + str(areas.min()) + " max: " + str(areas.max()))
    # polar = []
    # cart = []
    # center = findFlowCenter(flows)
    # for f in flows:
    #     polar.append(getFlowPolarEntropy(f))
    #     cart.append(getFlowCartEntropy(f))
    # polar = np.array(polar)
    # cart = np.array(cart)
    # print("Polar Entropy mean, ", polar.mean(axis=0),
    #       "Polar Entropy std: ", polar.std(axis=0))
    # print("Cartesian Entropy mean, ", cart.mean(axis=0),
    #       "Cartesian Entropy std: ", cart.std(axis=0))
    # print("Mean Translation error: " + str(getTranslationFlowError(flows).mean()))
    # print("Rotation correlations: ")
    # print(getRotationCorrelations(flows).mean(axis=0))
    # plt.figure("base image")
    # plt.imshow(imgs[0])
    # plt.show(block=False)
    # plt.figure("new image")
    # plt.imshow(imgs[i])
    # plt.show(block=False)
    # plt.figure("dis")
    # plt.imshow(flow_vis.flow_to_color(flow, convert_to_bgr=True))
    # plt.show()
