#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
from pynput import keyboard


saved = False
save_time = False

def Imagecallback(msg):
    global saved, save_time
    img = cv2.imdecode(np.fromstring(msg.data, dtype=np.uint8), 1)
    img_time = str(time.time())
    if saved is False:
        cv2.imwrite("/home/yipai/imgs/" + img_time + ".jpg", img)
        # cv2.imshow('CompressedImage', img)
        # cv2.waitKey(1000)
        # saved=True
    if save_time:
        cv2.imwrite("/home/yipai/imgs_times/" + img_time + ".jpg", img)
        rospy.loginfo("One image time saved, press Esc to exit")
        save_time = False


def on_press(key):
    pass
    # if key == keyboard.Key.right:
        # global save_time = True
        # print("right arrow pressed")


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        rospy.signal_shutdown("Manually shut down")
        return False
    elif key == keyboard.Key.right:
        global save_time
        save_time = True


def main():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('image_saver', anonymous=True, disable_signals=True)

    rospy.Subscriber("raspicam_node_l/image/compressed",
                     CompressedImage, Imagecallback)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    # cv2.waitKey(50000)


if __name__ == '__main__':
    main()
