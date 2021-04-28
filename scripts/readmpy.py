import numpy as np
import matplotlib.pyplot as plt

txt1 = "/home/zgl/Desktop/result/9_11/predict/center_x_list.npy"
txt2 = "/home/zgl/Desktop/result/9_11/predict/predict_list.npy"
txt3 = "/home/zgl/Desktop/result/9_11/predict/time.npy"
txt4  ="/home/zgl/Desktop/result/9_11/predict/rot_list.npy"

otxt1 = "/home/zgl/Desktop/result/9_9/mpu_on_foot/center_x_list_nc2.npy"
otxt2 = "/home/zgl/Desktop/result/9_9/mpu_on_foot/rot_list_nc2.npy"
otxt3 = "/home/zgl/Desktop/result/9_9/mpu_on_foot/time_nc2.npy"


w1 = np.load(txt1)
w2 = np.load(txt2)
w3 = np.load(txt3)
w4 = np.load(txt4)
w4 = w4 + 2
# o1 = np.load(otxt1)
# o2 = np.load(otxt2)
# o3 = np.load(otxt3)

t_end = w3[0]

l = len(w1)
 
t = np.zeros((l,))
# print(t)
# print(t[0])
for i in range(0, l):
    t[i] = i * t_end/l
    
# ot_end = o3[0]

# l = len(o1)
 
# ot = np.zeros((l,))
# print(t)
# print(t[0])
# for i in range(0, l):
#     ot[i] = i * ot_end/l

fig, axs = plt.subplots(3)
fig.suptitle('subplots')
axs[0].plot(t, w1)
axs[1].plot(t, w2)
axs[2].plot(t, w4)
# axs[3].plot(t, o2)
plt.savefig('/home/zgl/Desktop/02.png')
plt.show()

# print(w1,w2,w3)
# print(len(w1),len(w2))