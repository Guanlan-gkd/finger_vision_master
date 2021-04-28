#! /usr/bin/env python3

import glob
import time

import cv2

from Fv_utils import Tracker, getDensity

if __name__ == "__main__":
    # folder_name = "/home/yipai/11_9/9_4/*.jpg"
    folder_name = "/home/yipai/data_11_27/3x3/img/*.jpg"
    print("folder: " + folder_name)
    file_paths = sorted(glob.glob(folder_name))

    # video writer
    # out = cv2.VideoWriter("/home/yipai//manualslipdepth.avi", cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(
    #     'M', 'J', 'P', 'G'), 10, (640, 480))

    flows = []
    m = []
    start_time = time.time()
    tracker = Tracker(isFast=True)
    for i in range(len(file_paths)):
        img = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        flow = tracker.track(frame)
        depth_image = getDensity(flow.astype('float32'))
        depth_image = cv2.resize(
            depth_image, (640, 480), interpolation=cv2.INTER_CUBIC)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_HOT)
        # out.write(depth_image)
        # cv2.imwrite("/home/yipai/depth_imgs/" +
        #             str(time.time()) + ".jpg", depth_image)

        # diff = np.hypot(flow[:, :, 0], flow[:, :, 1])
        # print(diff.max())
        # diff = (diff / 10 * 255).astype("uint8")
        # diff = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
        # out.write(diff)
        # plt.clf()
        # # plt.hist(diff.flatten(), bins=256)
        # plt.imshow(diff)
        # plt.savefig("/home/yipai/flows/" + str(time.time()) + ".jpg")

        # flows.append(flow)
        # cropped_flow = flow[60:-60, 80:-80]
        # cropped_flow = deepcopy(flow)
        # cropped_flow[180-80:180+80, ]

        # flow_on_image = put_optical_flow_arrows_on_image(img, flow)
        # cv2.imwrite("/home/yipai/flows/" +
        # str(time.time()) + ".jpg", flow_on_image)
        # flow_on_image = np.hypot(decomposer.d[:, :, 0], decomposer.d[:, :, 1])
        # flow_on_image = np.sort(flow_on_image.flatten())
        # flow_on_image = flow_on_image[-20:].mean()
        # area_points.append(flow_on_image)
        # flow_on_image = (flow_on_image/15*255).astype(dtype='uint8')
        # flow_on_image = cv2.applyColorMap(flow_on_image, cv2.COLORMAP_HOT)
    # # flow_on_image = cv2.resize(flow_on_image, (flow_on_image.shape[1], flow_on_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite("/home/yipai/flows/" +
    #             str(time.time()) + ".jpg", flow_on_image)

        # cropped_flow = flow[60:-60, 80:-80]
        # cv2.imwrite("/home/yipai/depth_imgs_times/" +
        #             str(time.time()) + ".jpg", area_image)
        # decomposer.decompose(cropped_flow, verbose=0)
        # flow_on_image = np.hypot(cropped_flow[:, :, 0], cropped_flow[:, :, 1])
        # flow_on_image = flow_on_image.flatten()
        # sorted_disp_mag = np.sort(flow_on_image)
        # flow_on_image = sorted_disp_mag[-100:].mean()
        # area_points.append(flow_on_image)
        # if i // (len(file_paths)//10) > (i-1) // (len(file_paths)//10):
        #     print("Finished " + str(i/float(len(file_paths))*100) + "%")
    print("Frequency: " + str(float(len(file_paths) - 1) /
                              (time.time()-start_time)) + " fps")
    # out.release()
    # density = getDensity(flow.astype('float32'))
    # depth_image = cv2.applyColorMap(density, cv2.COLORMAP_HOT)
    # cv2.imwrite("/home/yipai/depth_imgs/" +
    #             str(time.time()) + ".jpg", depth_image)
    # x, y = density.shape
    # x = np.arange(0, x)
    # y = np.arange(0, y)
    # X, Y = np.meshgrid(x, y)
    # fig = plt.figure()
    # ax = fig.axes(projection='3d')
    # ax.plot_surface(X, Y, density)
    # plt.show()
    # savemat("/home/yipai/depth.mat", mdict=(dict(depth=depth_image)))
