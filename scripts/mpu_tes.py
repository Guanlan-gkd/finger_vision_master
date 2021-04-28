#!/usr/bin/env python
import math  
import numpy as np
import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import String
from time import sleep
from scipy import signal
data = np.genfromtxt(fname='/home/zgl/Desktop/xr1.txt');
data = data[1500:1600]
Wn = 4/214
b, a = signal.butter(2, Wn, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
r1 = np.array([1,2])
r2 = np.array([3,4])

r3 = np.concatenate((r1,r2),axis = None)


print(r3)

filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
plt.plot(filtedData)
plt.plot(data)
plt.show()
rospy.init_node('flow_visualization_listener', anonymous=True)
rot = []

def GetRotation(data):
    global rot
    l = data.data
    li = list(l.split(","))
    rot = [float(li[0]),float(li[1])]
    print(rot)
    
def listener():
    rospy.Subscriber("rotation", String, GetRotation, queue_size=10)
    
    rospy.spin()
    
if __name__ == '__main__':
    listener()

