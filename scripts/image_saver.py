#!/usr/bin/env python
import math  
import cv2
import numpy as np
import rospy
# import gpiozero
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from pynput import keyboard
from fv_lib.Fv_utils import Tracker
from time import sleep
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pickle
from scipy import spatial
from scipy.special import softmax
# from gpiozero.pins.pigpio import PiGPIOFactory
# import gpiozero

# factory = PiGPIOFactory(host='192.168.1.104')
deg_board = -12.18
point_cont = 558
flow = []
image1 = []
mtx = np.load('/home/zgl/Desktop/old_KD/mtx.npy')
dist = np.load('/home/zgl/Desktop/old_KD/dist.npy')
## global variables
tracker = Tracker(isFast=True)
# base_frame = cv2.imread(
#     "/home/yipai/1zi/1573025936.9.jpg", cv2.IMREAD_GRAYSCALE)
rospy.init_node('flow_visualization_listener', anonymous=True)

tick_time = rospy.Time.now()
frame_counter = 0

rot = []
degree = []

# J1 = gpiozero.PWMOutputDevice(4, pin_factory=factory, frequency=50, initial_value=4/100)
# J2 = gpiozero.PWMOutputDevice(17, pin_factory=factory, frequency=50, initial_value=7.5/100)


def GetRotation(data):
    global rot
    l = data.data
    li = list(l.split(","))
    rot = [float(li[0]),float(li[1])]
    # print(rot)

def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=1.2):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow_image *40

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, scaled_flow.shape[1], 30), range(0, scaled_flow.shape[0], 30)), 2)
    flow_end = (scaled_flow[flow_start[:, :, 1],
                            flow_start[:, :, 0], :] + flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(
        scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :], axis=2)
    norm[norm < threshold] = 0
    # Draw all the nonzero values
    nz = np.nonzero(norm) 

    # print(norm.max())
    norm = np.asarray(norm / 150.0*255.0, dtype='uint8')
    # print(norm.max())
    color_image = cv2.applyColorMap(norm, cv2.COLORMAP_RAINBOW).astype('int')
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(0,255,0),
                        thickness=2,
                        tipLength=.2)
    return image


def Imagecallback(msg):
    global tracker, frame_counter, tick_time, tree, point_3d, rot, flow
    global image1
    
    # print(float((rospy.Time.now()-msg.header.stamp).nsecs)/1e9)
    # if (rospy.Time.now()-msg.header.stamp).nsecs > 1e8:
    #     print("throw one frame")
    
    #     return
  
    
    img = cv2.imdecode(np.fromstring(
        msg.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    x,y,w,h = roi
    # print(w,h)
    img = dst[y:y+h, x:x+w]
    (l,w) = img.shape
    img = img[int(0.1*l):int(0.9*l),int(0.1*w):int(0.8*w)]
    # cv2.imshow("img",img)    
    (l,w) = img.shape
    s_w = w
    s_l = l
    img = cv2.resize(img, (s_w, s_l))
    image1 = img
    flow = tracker.track(img)
    # print(flow.max())
    # s = np.sum(flow,axis = 1)/s_w
    # s = np.sum(s,axis = 0)/s_l
    # print(s)
    # mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
    # mag = (mag/2.5 * 255).astype('uint8')
    # (l,w) =  mag.shape
    # nmag = mag[int(0.1*l):int(0.9*l),int(0.1*w):int(0.8*w)]
    # print(flow.shape)
    # print(mag.shape)

    # mag = (mag > 0.2*127) * mag
    # imag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)

    # cv2.imshow("imag",imag)
    # print(imag.max())
    # arrows = put_optical_flow_arrows_on_image(
        # image1, flow)
    # cv2.imshow('arrows',arrows)
    # cv2.waitKey()
    frame_counter += 1
    if rospy.Time.now() - tick_time > rospy.Duration(secs=10.0):
        freq = frame_counter / (rospy.Time.now()-tick_time).secs
        frame_counter = 0
        rospy.loginfo("frame rate: " + str(freq))
        # mag = np.hypot(flow[:, :, 0], flow[:, :, 1]).max()
        # rospy.loginfo("mag: " + str(mag))
        tick_time = rospy.Time.now()
        
        
def on_press(key):
    pass


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        rospy.signal_shutdown("Manually shut down")
        return False
    elif key == keyboard.Key.right:
        global tracker
        tracker.reset()
        rospy.loginfo("Frame reset")
    elif key == keyboard.Key.left:
        global flow
        global rot
        # global image1
        # # global point
        # # global centerl
        global point_cont
        global deg_board
        # degree.append(rot)
        # print('saved' + str(centerl))
        # point.append(centerl)
        # print(np.asarray(point))
        # print(point_cont)
        # np.save('/home/zgl/Desktop/ct2/centerl{}.npy'.format(point_cont), np.asarray(centerl))
        np.save('/home/zgl/Desktop/data_10_16/deg_mpu/degree_{}.npy'.format(point_cont), np.asarray(rot))
        # print(degree)
        np.save('/home/zgl/Desktop/data_10_16/flow/flow_{}.npy'.format(point_cont), np.asarray(flow))
        
        np.save('/home/zgl/Desktop/data_10_16/deg_board/deg_b_{}.npy'.format(point_cont), np.asarray(deg_board))
        # print(flow.shape)
        # np.save('/home/zgl/Desktop/conp/point{}.npy'.format(point_cont), np.asarray(point))
        # cv2.imwrite('/home/zgl/Desktop/result_point/image2.jpg',image1)
        print('saved',rot, deg_board, point_cont, flow.max())
        point_cont = point_cont + 1
    # elif key == keyboard.Key.down:
        
    #     point=point[:-1]
    #     np.save('/home/zgl/Desktop/point.npy', np.asarray(point))

def listener():

    rospy.Subscriber("raspicam_node/image/compressed",
                     CompressedImage, Imagecallback, queue_size=1, buff_size=2**22)
    # rospy.Subscriber("raspicam_node_l/image/compressed",
    #                  CompressedImage, Imagecallback, queue_size=1)
    rospy.Subscriber("rotation", String, GetRotation, queue_size=10)
    print()
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    print("m0")
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    print('m1')


if __name__ == '__main__':
    listener()
