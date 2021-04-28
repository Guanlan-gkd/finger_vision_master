#!/usr/bin/env python
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
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


regr_x = pickle.load(open('/home/zgl/Desktop/trained_regr/regr_x.p', 'rb')) 
regr_y = pickle.load(open('/home/zgl/Desktop/trained_regr/regr_y.p', 'rb'))


point_cont = 0
image1 = np.zeros((512,384,3))
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

def GetRotation(data):
    global rot
    l = data.data
    li = list(l.split(","))
    rot = [float(li[0]),float(li[1])]
    

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

def drawcenter(nmag):
    m_center = rospy.Publisher('momment_center', String, queue_size=10)
    inmag = nmag
    nmag = cv2.cvtColor(nmag, cv2.COLOR_BGR2GRAY)
    nmag[nmag <= 0.7*nmag.max()] = 0
    M = cv2.moments(nmag)
    # print(int(M['m00'])/1000)
    if int(M['m00']) > 1000000:
        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = int(nmag.shape[1]/2)
        cY = int(nmag.shape[0]/2)
    cent = str(cX) + "," + str(cY)
    m_center.publish(cent)
    print(cX,cY)
    # print(nmag.shape)
    cv2.circle(inmag, (cX, cY), 5, (255, 0, 0), 5)
    
    return inmag    

def Imagecallback(msg):
    global tracker, frame_counter, tick_time, tree, point_3d, rot
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
    cv2.imshow("img",img)    
    (l,w) = img.shape
    s_w = w
    s_l = l
    img = cv2.resize(img, (s_w, s_l))
    image1 = img
    flow = tracker.track(img)
    oflow = flow
    scale_factor = 4
    scale_shape = ((flow.shape[1]/scale_factor),int(flow.shape[0]/scale_factor))
    # print(scale_shape)
    flow = cv2.resize(flow,scale_shape)
    flow = flow.reshape((1,-1))
    # print(flow.shape)
    # s = np.sum(flow,axis = 1)/s_w
    # s = np.sum(s,axis = 0)/s_l
    # print(s)
    mag = np.hypot(oflow[:, :, 0], oflow[:, :, 1])
    mag = (mag/2.5 * 255).astype('uint8')
    (l,w) =  mag.shape
    nmag = mag[int(0.1*l):int(0.9*l),int(0.1*w):int(0.8*w)]
    # print(flow.shape)
    # print(mag.shape)

    mag = (mag > 0.2*127) * mag
    nmag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
    # print(nmag.size)
    # inmag = drawcenter(nmag)
    y_predict_x = regr_x.predict(flow)
    y_predict_y = regr_y.predict(flow)
    print(float(y_predict_x[0]), float(y_predict_y[0]))
    print(rot)
    print("---------------------------------------------")
    sleep(0.5)
    # cv2.imshow("imag",inmag)
    cv2.imshow("imag",nmag)
    # arrows = put_optical_flow_arrows_on_image(
    #     image1, flow)
    # cv2.imshow('arrows',arrows)
    cv2.waitKey(1)
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
        # global image1
        global point
        # global centerl
        global point_cont
        point_cont = point_cont + 1
        
        print('saved' + str(point))
        # point.append(centerl)
        # print(np.asarray(point))
        # print(point_cont)
        # np.save('/home/zgl/Desktop/ct2/centerl{}.npy'.format(point_cont), np.asarray(centerl))
        # np.save('/home/zgl/Desktop/conp/point{}.npy'.format(point_cont), np.asarray(point))
        # cv2.imwrite('/home/zgl/Desktop/result_point/image2.jpg',image1)
        print('saved')
    # elif key == keyboard.Key.down:
        
    #     point=point[:-1]
    #     np.save('/home/zgl/Desktop/point.npy', np.asarray(point))

def listener():

    rospy.Subscriber("raspicam_node/image/compressed",
                     CompressedImage, Imagecallback, queue_size=1, buff_size=2**22)
    # rospy.Subscriber("raspicam_node_l/image/compressed",
    #                  CompressedImage, Imagecallback, queue_size=1)
    rospy.Subscriber("rotation", String, GetRotation, queue_size=10)
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
