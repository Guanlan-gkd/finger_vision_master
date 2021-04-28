import glob
import os
import time
from copy import deepcopy

import cv2
import freud
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from Fv_utils import Tracker, getDensity


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


def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=0.0):
    # Don't affect original image
    image = image.copy()
    optical_flow_image = (optical_flow_image*5).copy()
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, optical_flow_image.shape[1], 20), range(0, optical_flow_image.shape[0], 20)), 2)
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


def flow_to_color(flow):
    mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
    # print(mag.max())
    mag = (mag/7.0*255).astype("uint8")
    phase = ((np.arctan2(flow[:, :, 1], flow[:, :, 0]
                         ) + np.pi)/2/np.pi*179).astype("uint8")
    value = np.ones_like(mag, dtype="uint8")*255
    colors = np.stack((phase, mag, value), axis=2)
    return cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":

    # folder_name = "/home/yipai/data_11_27/3x3/times/*.jpg"
    folder_name = "/home/yipai/calibration_imgs/*.jpg"
    print("folder: " + folder_name)
    file_paths = sorted(glob.glob(folder_name))
    img_names = getImageNames(folder_name)
    tik_folder_name = "/home/yipai/calibration_times/*.jpg"
    tik_img_names = getImageNames(tik_folder_name)
    tik_img_names = tik_img_names[1:]
    
    # DIS method
    tracker = Tracker(isFast=True)

    areas = []
    area_points = []
    for i in range(len(file_paths)):
        img = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        flow = tracker.track(frame)
        
        if img_names[i] == tik_img_names[0]:
            # cropped_flow = flow[60:-60, 80:-80]
            cropped_flow = deepcopy(flow)
            # area_image = getVoronoiAreas(decomposer.d)
            area_image = getDensity(cropped_flow)
            # flow_on_image = put_optical_flow_arrows_on_image(img, flow)
            # flow_on_image = np.hstack((flow_on_image1, edge, flow_on_image2, edge, flow_on_image3))
    
            # cv2.imwrite("/home/yipai/flows/" +
            #         str(time.time()) + ".jpg", flow_on_image)

            cv2.imwrite("/home/yipai/depth_imgs/" +
                        str(time.time()) + ".jpg", cv2.applyColorMap(area_image, cv2.COLORMAP_HOT))
            area_image = np.sort(area_image.flatten())[-7:-4].mean()
            area_points.append(area_image)

            tik_img_names.pop(0)
            if not tik_img_names:
                break
        if i % (len(file_paths)//10) == 0:
            print("Finished " + str(100*i/float(len(file_paths))) + "%")
            
    
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

    # areas = np.array(areas).transpose((2, 0, 1))
    # savemat("/home/yipai/data.mat", mdict=(dict(areas=areas,
    #                                             depth_points=depth_points, area_points=area_points)))
    # m = np.array(m)
    # print("min: " + str(areas.min()) + " max: " + str(areas.max()))
