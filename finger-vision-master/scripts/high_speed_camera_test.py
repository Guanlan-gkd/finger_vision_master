import cv2
import time
import numpy as np
# mtx = np.load('/home/zgl/Desktop/mtx/mtx_h.npy')
# dist = np.load('/home/zgl/Desktop/mtx/dist_h.npy')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

from pynput import keyboard

# print(cap.isOpened())
# for i in range(0,10):
#     ret, frame = cap.read()
#     cv2.imwrite('/home/zgl/Desktop/calibdata/'+str(i)+'.jpg',frame)
#     cv2.imshow("test", frame)
#     cv2.waitKey()
#     print("saved")
# print("s")    
# cap.release()
# cv2.destroyAllWindows()

t1 = time.time()
while True:
    ret, frame = cap.read()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("test", frame)
    # h,  w = frame.shape[:2]
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))   
    # undistort6
    # dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    if ret:
        cv2.imshow('calibresult',frame)
        print(1.0/(time.time()-t1))
        t1 = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

