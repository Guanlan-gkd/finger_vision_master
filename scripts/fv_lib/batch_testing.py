#! /usr/bin/env python3

import glob
import os
import time

import cv2
import matplotlib.pyplot as plt

from Fv_utils import Tracker, getDensity

if __name__ == "__main__":
    folder_name = "/home/yipai/depth_shape_data/data_source"
    output_folder = "/home/yipai/depth_shape_data/result"
    print("folder: " + folder_name)
    # 

    # video writer
    # out = cv2.VideoWriter("/home/yipai//manualslipdepth.avi", cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(
    #     'M', 'J', 'P', 'G'), 10, (640, 480))
    for inputFolderName in os.listdir(folder_name):
        file_paths = os.path.join(folder_name, inputFolderName, "*.jpg")
        print("Generating for input:" + file_paths)
        file_paths = sorted(glob.glob(file_paths))
        output_path = os.path.join(output_folder, inputFolderName)
        os.mkdir(output_path)
        tracker = Tracker(isFast=True)
        start_time = time.time()
        for i in range(len(file_paths)):
            img = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            flow = tracker.track(frame)
            depth_image = getDensity(flow.astype('float32'))
            # depth_image = cv2.resize(
            #     depth_image, (640, 480), interpolation=cv2.INTER_CUBIC)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_HOT)
            
            output_name = os.path.join(output_path, os.path.basename(file_paths[i]))
            cv2.imwrite(output_name, depth_image)
            # plt.cla()
            # plt.hist(depth_image.flatten(), bins=80, range=(0, 80))
            # plt.savefig(output_name)

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
