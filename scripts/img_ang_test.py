#!/usr/bin/env python
import math  
import cv2
import numpy as np
import rospy
import time
# import gpiozero
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from pynput import keyboard
from fv_lib.Fv_utils import Tracker

# import tensorflow
# import os
# import matplotlib.pyplot as plt
# print(tensorflow.__version__)
# import keras
# print(keras.__version__)
# from keras import regularizers
# from keras import initializers
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense 
# from keras.layers import Dropout

from scipy import signal
import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt



print("import done")


# 0 for GPU, -1 for CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# config = tensorflow.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# tensorflow.compat.v1.keras.backend.set_session(tensorflow.compat.v1.Session(config=config))

on_off = 0
point_cont = 0
point_pair = []
pred = []
image1 = np.zeros((512,384,3))
mtx = np.load('/home/zgl/Desktop/old_KD/mtx.npy')
dist = np.load('/home/zgl/Desktop/old_KD/dist.npy')
## global variables
tracker = Tracker(isFast=True)
# base_frame = cv2.imread(
#     "/home/yipai/1zi/1573025936.9.jpg", cv2.IMREAD_GRAYSCALE)
rospy.init_node('flow_visualization_listener', anonymous=True)
pub_pre = rospy.Publisher('prediction', String, queue_size=10)
tick_time = rospy.Time.now()
frame_counter = 0
rot = []
rot_list = []
center_x_list = []
center_x_list_record = 0
num_sample = 252
scale_factor = 1
time_init = 0

sample_flow = np.load('/home/zgl/Desktop/flow/flow_1.npy')

sample_deg = np.load('/home/zgl/Desktop/deg_neo/deg_1.npy').reshape((1,2))

scale_shape = (int(sample_flow.shape[1]/scale_factor),int(sample_flow.shape[0]/scale_factor))
print("scale shape",scale_shape)

# model = Sequential()  
# model.add(Conv2D(64, kernel_size = (3,3), strides = 1, input_shape = (scale_shape[1],scale_shape[0], 2), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Conv2D(32, kernel_size = (3,3), strides = 1, activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Conv2D(32, kernel_size = (3,3), strides = 1, activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))



# # model.add(Conv2D(32, 3, 1, activation = 'relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))

# # full connected
# model.add(Flatten())

# model.add(Dense(units = 128, activation = 'relu',
#                 kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                 bias_regularizer = regularizers.l2(1e-4),
#                 activity_regularizer = regularizers.l2(1e-5),
#                 kernel_initializer = initializers.RandomNormal(stddev=0.01),
#                 bias_initializer = initializers.Zeros()
#                 ))

# model.add(Dropout(rate = 0.3))


# # model.add(Dense(output_dim = 10, activation = 'relu'))
# # output is 2 degree
# model.add(Dense(units= sample_deg.shape[1], activation = 'linear',
#                 kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                 bias_regularizer = regularizers.l2(1e-4),
#                 activity_regularizer = regularizers.l2(1e-5),
#                 kernel_initializer = initializers.RandomNormal(stddev=0.01),
#                 bias_initializer = initializers.Zeros()
#                 ))



# #compile, loss funtion is MSE
# opt = keras.optimizers.Adam(lr = 1e-4)
# # model.compile(optimizer = opt, loss = 'mean_squared_error', metrics=['rmse'])



# model.compile(
#     optimizer=opt,
#     loss='mean_squared_error',
#     metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])

# print(model.summary())

# model.load_weights("/home/zgl/Desktop/trained model/Aug14_0_weights")
# print(sample_flow.shape)
# pred1 = model.predict(np.array([sample_flow]))
# print(pred1)
# # test on whole set

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
    # print(cX,cY)
    # print(nmag.shape)
    cv2.circle(inmag, (cX, cY), 5, (255, 0, 0), 5)
    
    return inmag    

def Imagecallback(msg):
    global tracker, frame_counter, tick_time, tree, point_3d, rot, sample_flow, pred, pub_pre
    global image1, center_x_list, rot_list
    
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
    flow_x = flow[:,:,0]
    # entr_img = entropy(flow_x, disk(10))
    # print(entr_img.shape)
    # img_uni = skimage.util.img_as_ubyte(entr_img)
    # cv2.imshow("", img_uni)
    # cv2.waitKey()
    # print(flow_x.max())
    # print(flow.shape)
    if on_off == 1:
        pass
        # pred = model.predict(np.array([flow]))
        # pred = model.predict(np.array([]))
        # pub_pre.publish(str(pred[0, 0]) + "," + str(pred[0, 1]))
        # print(pred, rot[1], pred[0, 0] - rot[1])
        # print(float(rot[1]))
    else:
        print("press left",flow.shape,sample_flow.shape)
    mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
    mag = (mag/2.5 * 255).astype('uint8')
    indices = np.where(mag == mag.max())
    # print(indices[1].shape)
    indices = np.mean(indices, axis=1)
    # print(indi.astype(int))
    # break
    # print(indices[0].shape)
    center_coordinates = (int(indices[1]), 93)
    if center_x_list_record == 1:
        center_x_list.append(int(indices[1]))
        rot_list.append(float(rot[1]))
    print(center_coordinates, flow.shape)
    # (l,w) =  mag.shape
    # imag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)

    # cv2.imshow("imag",imag)

    # cv2.waitKey(1)
    frame_counter += 1
    if rospy.Time.now() - tick_time > rospy.Duration(secs=10.0):
        freq = frame_counter / (rospy.Time.now()-tick_time).secs
        frame_counter = 0
        rospy.loginfo("frame rate: " + str(freq))
        # mag = np.hypot(flow[:, :, 0], flow[:, :, 1]).max()
        # rospy.loginfo("mag: " + str(mag))
        tick_time = rospy.Time.now()
    time.sleep(0.1)
        
        
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
        global on_off
        if on_off ==0:
            on_off = 1
        # global image1
        # global point
        # global centerl
        print('start predict')
    elif key == keyboard.Key.ctrl_r:
        global point_cont, point_pair, pred, rot,center_x_list_record, center_x_list, rot_list, time_init
        if center_x_list_record == 0:
            time_init = time.time()
            center_x_list_record = 1
            print("start record data")
        elif center_x_list_record == 1:
            center_x_list_record = 0
            print(center_x_list)
            print(rot_list)
            txt1 = "/home/zgl/Desktop/result/9_9/mpu_on_foot/center_x_list_2.npy"
            txt2 = "/home/zgl/Desktop/result/9_9/mpu_on_foot/rot_list_2.npy"
            txt3 = "/home/zgl/Desktop/result/9_9/mpu_on_foot/time_2.npy"
            np.save(txt1, np.asarray(center_x_list))
            np.save(txt2, np.asarray(rot_list))
            np.save(txt3, np.asarray([time.time()-time_init]))
            print("stop record data")
        # ro = pred[0, 0]
        # print([ro,float(rot[1])])
        # point_pair.append([ro,float(rot[1])])
        # print(point_pair)
        # print(np.asarray(point_pair).shape)
        # point_cont = point_cont + 1
        # np.save('/home/zgl/Desktop/point_pair.npy', np.asarray(point_pair))
        # print('saved' + str(point))
        # point.append(centerl)
        # print(np.asarray(point))
        # print(point_cont)
        # np.save('/home/zgl/Desktop/ct2/centerl{}.npy'.format(point_cont), np.asarray(centerl))
        # np.save('/home/zgl/Desktop/conp/point{}.npy'.format(point_cont), np.asarray(point))
        # cv2.imwrite('/home/zgl/Desktop/result_point/image2.jpg',image1)
        
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
