from nHHD import *
import numpy as np 
import cv2

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

# from undistort_crop_resize import *

from sklearn.neighbors import KDTree

import time

from scipy import ndimage
from skimage import morphology, util, filters
import skimage

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

def crop(img):
    w = 720
    h = 1280
    dw = 50
    dh = 300
    y_min = dw
    y_max = w - dw
    x_min = dh -50
    x_max = h - dh
        
    return img[y_min:y_max, x_min:x_max].copy()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()


params.filterByColor = 0
params.blobColor = 0
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 60

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)



def blob_detect(img):
    keypoints = detector.detect(img)

    # print(len(keypoints))
    locs = []
    for i in range(0, len(keypoints)):
        locs.append([keypoints[i].pt[0], keypoints[i].pt[1] ] )

    # print(np.array(locs))
    return np.array(locs), keypoints



count = 0

loc_0 = []

t1 = time.time()
while (True):
    # Capture frame-by-frame
    # print('testing')
    ret, frame = cap.read()

    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    
    gray_crop = crop(gray)
    gray_crop = cv2.resize(gray_crop, (int(gray_crop.shape[1]/1.5), int(gray_crop.shape[0]/1.5)))
    loc, keypoints = blob_detect(gray_crop)
    # print(loc.shape)
    # print(loc)
    im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]),
                                          (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('keypoints', im_with_keypoints)

    if count == 0:

        loc_0 = loc.copy()
        recent_loc = loc.copy()
    elif count > 0:
        
        # print(loc_0[1,:])
        kdt = KDTree(loc, leaf_size=30, metric='euclidean')
        dist, ind = kdt.query(recent_loc, k=1)
        # kd tree matched points that are too far away are ignored
        thd = (dist < 14)*1
        thd_nz = np.where(thd)[0]
        # update point if close enough point are detected
        recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))

        # visualize the displacement field
        loc_v = 2*recent_loc - loc_0  # diff vector

        # decompose vector field
        # field = nHHD()
        # field.decompose(loc_v)  
        
        print(int(1.0/(time.time()- t1)))
        t1 = time.time()
        
        
        for i in range(0, len(loc_0)):
            cv2.arrowedLine(gray_crop, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
                            (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 0, 255), thickness=2)
        cv2.imshow('arrow', gray_crop)

        # df = pd.DataFrame(np.concatenate((recent_loc, loc_v), axis=1), columns=['x', 'y', 'xt', 'yt'])


    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
