#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
from fv_lib import flow_vis
from fv_lib.pynhhd.nHHD import nHHD
def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=2.0):
    # Don't affect original image
    image = image.copy()

    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, optical_flow_image.shape[1], 30), range(0, optical_flow_image.shape[0], 30)), 2)
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
                        thickness=2,
                        tipLength=.2)
    return image


## global variables
dis_inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
dis_inst.setFinestScale(3)  # 2 as default
dis_inst.setGradientDescentIterations(12)  # 12 as default
dis_inst.setVariationalRefinementIterations(5)  # 0 as default
# dis_inst.setPatchSize(12)  # 8 as default
base_frame = None
prev_time = time.time()
flow = None
velocity = None
prev_flow = None
frame_counter = 0

# decomp object
dx = float(1.0)
dy = float(1.0)
grid = (480, 640)
decomposer = nHHD(grid=grid, spacings=(dy, dx))
def Imagecallback(msg):
    global dis_inst, base_frame, prev_time, flow, velocity, prev_flow, frame_counter, decomposer
    img = cv2.imdecode(np.fromstring(msg.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if base_frame is None:
        base_frame = img
        prev_time = time.time()
        return
    else:
        if velocity is None:
            flow = dis_inst.calc(base_frame, img, flow)
        else:
            flow = dis_inst.calc(base_frame, img, flow + velocity)
        if prev_flow is not None:
            velocity = flow - prev_flow
        prev_flow = flow
    # cv2.imshow("image", img)
    decomposer.decompose(flow, verbose=0)
    cv2.imshow("arrows", put_optical_flow_arrows_on_image(img, decomposer.d))   
    # cv2.imshow("flow color", flow_vis.flow_to_color(flow, convert_to_bgr=True))
    cv2.waitKey(1)
    frame_counter += 1
    if time.time() - prev_time > 2:
        freq = frame_counter / (time.time()-prev_time)
        frame_counter = 0
        rospy.loginfo("frame rate: " + str(freq))
        mag = np.square(flow).sum()
        rospy.loginfo("mag: " + str(mag))
        prev_time = time.time()


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('flow_visualization_listener', anonymous=True)

    rospy.Subscriber("raspicam_node_l/image/compressed", CompressedImage, Imagecallback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    # cv2.waitKey(50000)


if __name__ == '__main__':
    listener()
