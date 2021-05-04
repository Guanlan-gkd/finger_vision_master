from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
from nHHD import *
import numpy as np 
import cv2
import matplotlib.pyplot as plt


# from undistort_crop_resize import *

from sklearn.neighbors import KDTree

import time

from scipy import ndimage
from skimage import morphology, util, filters
from scipy.interpolate import griddata
import skimage

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

def crop(img):
    w = 600
    h = 800
    dw = 20
    dh = 40
    y_min = dw
    y_max = w - dw
    x_min = dh
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
params.minArea = 10
params.maxArea = 60

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4

# Filter by Inertia
params.filterByInertia = False
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
grid_x, grid_y = np.mgrid[120:368:18, 85:335:18]
x_axis = grid_x.reshape(-1).astype(np.int64)
y_axis = grid_y.reshape(-1).astype(np.int64)
print(grid_y.shape)

t1 = time.time()
while (True):
    # Capture frame-by-frame
    # print('testing')
    ret, frame = cap.read()

    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    
    gray_crop = crop(gray)
    # gray_crop = gray
   
    gray_crop = cv2.resize(gray_crop, (int(gray_crop.shape[1]/1.5), int(gray_crop.shape[0]/1.5)))
    # print("gray_crop shape: ", gray_crop.shape)
    loc, keypoints = blob_detect(gray_crop)
    # print(loc.shape)
    # print(loc)
    im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]),
                                          (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('keypoints', im_with_keypoints)
    # print("loc shape", loc.shape)

    # fig, ax = plt.subplots()
    # ax.imshow(im_with_keypoints)
    # plt.show()

    if count == 0:
        x_thr = 0.6
        y_thr = 0.6
        loc_0 = loc.copy()    
        
        rm_index  =[]
        for i in range(0,loc_0.shape[0]):
            # print(loc_0[i])
            if loc_0[i, 1] < 60 or loc_0[i, 1] > 335 or loc_0[i, 0] < 100 or loc_0[i, 0] > 388:
                rm_index.append(i)
            # else:
            #     im_with_keypoints_new = cv2.circle(gray_crop, (int(loc_0[i,0]), int(loc_0[i,1])), 3, (255,0,0), 2)
        loc = np.delete(loc_0, rm_index, 0)
        loc_0 = loc.copy() 
        # print(loc_0.shape)
        # fig, ax = plt.subplots()
        # ax.imshow(im_with_keypoints_new)
        # plt.show()
        
        recent_loc = loc.copy()
        print(recent_loc.shape)
    elif count > 0:
        
        # print(loc_0[1,:])
        kdt = KDTree(loc, leaf_size=30, metric='euclidean')
        dist, ind = kdt.query(recent_loc, k=1)
        # kd tree matched points that are too far away are ignored
        thd = (dist < 16)*1
        thd_nz = np.where(thd)[0]
        # update point if close enough point are detected
        recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))

        # visualize the displacement field
        loc_v = recent_loc - loc_0  # diff vector
        # loc_v_draw = 2*recent_loc - loc_0
        
        # '''
        # interpolate diff vector into grid vector field

        # grid_vector = griddata(points, values, (grid_x, grid_y), method='nearest') 
        grid_vector =  griddata(recent_loc, loc_v, (grid_x, grid_y), method='cubic')
        grid_vector[np.isnan(grid_vector)] = 0
        

        # print(grid_vector.min(), grid_vector.max())
        grid_flat = grid_vector.reshape(-1,2).astype(np.int64)
        print(grid_vector.shape)
        

        # decompose vector field
        decompose_obj = nHHD(grid = (grid_vector.shape[0], grid_vector.shape[1]), spacings = (18, 18))
        decompose_obj.get_force(grid_vector, verbose = 0)
        # decompose_obj.decompose(grid_vector, verbose = 1)
          
        # '''
        
        print(int(1.0/(time.time()- t1)))
        t1 = time.time()
        
        # print(len(loc_0))
        # for i in range(0, len(loc_0)):
        #     cv2.arrowedLine(gray_crop, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
        #                     (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 255, 255), thickness=2)
        # cv2.imshow('arrow', gray_crop)
        
        for i in range(0, grid_flat.shape[0]):
            # print((x_axis[i],y_axis[i]),((grid_flat[i, 0]), (grid_flat[i,1])))
            cv2.arrowedLine(gray_crop, (x_axis[i],y_axis[i]),
                            ((grid_flat[i, 0] + x_axis[i]), (grid_flat[i, 1] + y_axis[i])), 
                            (0, 0, 255), thickness=2)
        cv2.imshow('arrow', gray_crop)

        # df = pd.DataFrame(np.concatenate((recent_loc, loc_v), axis=1), columns=['x', 'y', 'xt', 'yt'])


    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(decompose_obj.Stor)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
