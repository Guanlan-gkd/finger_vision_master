#!/usr/bin/env python
import math  
import cv2
import numpy as np

from fv_lib.Fv_utils import Tracker
from time import sleep

from pynput import keyboard

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

tracker = Tracker(isFast=True)
switch = 1

def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=1.2):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow_image *4

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

def imgprocess():
    
    global tracker, cap, switch
    while switch:
        ret, frame = cap.read()
        # cv2.imshow('calibresult',frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # (l,w) = img.shape
        # img = img[int(0.1*l):int(0.9*l),int(0.2*w):int(0.7*w)]
        # img = cv2.resize(img, (int(0.5*w/2), int(0.8*l/2)))
        
        
        # cv2.imshow("1",img)   

        flow = tracker.track(img)
    
        
        # mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
        # mag = (mag/2.5 * 255).astype('uint8')
        # (l,w) =  mag.shape
        # nmag = mag[int(0.1*l):int(0.9*l),int(0.1*w):int(0.8*w)]
        # # # # print(flow.shape)
        # # # # print(mag.shape)

        # mag = (mag > 0.2*127) * mag
        # imag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
        # # print(mag.max())
w        
        
        # cv2.imshow("2",imag)
        arrows = put_optical_flow_arrows_on_image(
            img, flow)
        cv2.imshow('arrows',arrows)
        # if img is not None:
        
        # cv2.destroyAllWindows()
        
        
        # image = cv2.imread('/home/zgl/Desktop/plot1.png')
        # print(image.shape)
        # cv2.imshow('image',image)
        cv2.waitKey(1)

def on_press(key):
    pass


def on_release(key):
    global switch
    if key == keyboard.Key.esc:
        print("kill")
        switch = 0
    elif key == keyboard.Key.right:
        global tracker
        tracker.reset()
        print("Frame reset")


        


def realtime():

    
        

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        imgprocess()

    # spin() simply keeps python from exiting until this node is stopped




if __name__ == '__main__':
    realtime()
