import cv2
import numpy as np
import rospy
from pynput import keyboard
from sensor_msgs.msg import CompressedImage

from fv_lib.Fv_utils import Tracker#, getDensity
image_raw = []
image_arrow = []
image_mag = []
pre_flow = []
f = []
mtx = np.load('/home/zgl/Desktop/old_KD/mtx.npy')
dist = np.load('/home/zgl/Desktop/old_KD/dist.npy')
n = 0
flow = []
w = 0
l = 0
def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=0.0):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow_image * 500

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, scaled_flow.shape[1], 160), range(0, scaled_flow.shape[0], 160)), 2)
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
                        color= (0, 0, 255),
                        thickness=12,
                        tipLength=.40)
    return image

def put_optical_flow_arrows_on_image2(image, optical_flow_image, threshold=0.0):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow_image * 500

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(
        range(0, scaled_flow.shape[1], 160), range(0, scaled_flow.shape[0], 160)), 2)
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
                        color= (255, 0, 0),
                        thickness=12,
                        tipLength=.40)
    return image

def flow_to_color(flow):
    # print(np.hypot(flow[:, :, 0], flow[:, :, 1]).max())
    mag = (np.hypot(flow[:, :, 0], flow[:, :, 1])/30*255).astype("uint8")
    # phase = ((np.arctan2(flow[:, :, 1], flow[:, :, 0]
    #                      ) + np.pi)/2/np.pi*179).astype("uint8")
    # saturation = np.ones_like(mag, dtype="uint8")*255
    # colors = np.stack((phase, mag, saturation), axis=2)
    # return cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)


# def arrowProcess(cropped_flow):
#     # global cropped_flow
#     shape = cropped_flow.shape
#     arrows = put_optical_flow_arrows_on_image(np.zeros((shape[0], shape[1], 3)), cropped_flow)
#     # cv2.imshow("arrows", arrows)


# def colorProcess(cropped_flow):
#     # global cropped_flow
#     colors = flow_vis.flow_to_color(cropped_flow, convert_to_bgr=True)
#     # cv2.imshow("flow color", colors)


# def contourProcess(cropped_flow):
#     global decomposer
#     decomposer.decompose(cropped_flow, verbose=0)


## global variables
tracker = Tracker(isFast=True,adaptive=False)
# base_frame = cv2.imread(
#     "/home/yipai/1zi/1573025936.9.jpg", cv2.IMREAD_GRAYSCALE)
rospy.init_node('flow_visualization_listener', anonymous=True)
tick_time = rospy.Time.now()
frame_counter = 0
flow_sum_tick=False
flow_sum_list=[]
# # decomp object
# dx = float(1.0)
# dy = float(1.0)
# grid = (45, 60)
# # grid = (480, 640)
# decomposer = nHHD(grid=grid, spacings=(dy, dx))


# box = freud.box.Box.square(2*grid[1])
# voro = freud.voronoi.Voronoi(box, grid[1])
# x = np.linspace(-(grid[1]-1) / 2,
#                     (grid[1]-1) / 2, grid[1])
# y = np.linspace(-(grid[0]-1) / 2,
#                 (grid[0]-1) / 2, grid[0])
# coordinate = np.array(np.meshgrid(x, y)).transpose((1, 2, 0))

def getBoundary(image):
    # area = cv2.GaussianBlur(area, ksize=(0, 0), sigmaX=10.0)
    shape = image.shape[:2]
    image = cv2.resize(
        image, (shape[1]/5, shape[0]/5), interpolation=cv2.INTER_AREA)

    image = cv2.medianBlur(image, ksize=5)
    image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=1.0)

    boundary = cv2.Canny(image, 70, 100)
    boundary = cv2.resize(
        boundary, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    morphShape = cv2.MORPH_RECT
    morphStructure = cv2.getStructuringElement(morphShape, ksize=(3, 3))
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, morphStructure)
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, morphStructure)
    # image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    # cv2.imshow("boundary", image)
    return cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)


def Imagecallback(msg):
    global tracker, frame_counter, tick_time , flow_sum_tick,flow_sum, flow
    global image_raw, image_arrow, image_mag, n
    
    if rospy.Time.now()-msg.header.stamp > rospy.Duration(secs=0.3):
        t = rospy.Time.now()-msg.header.stamp
        #print(t)
        #print("throw one frame")
        return
    img = cv2.imdecode(np.fromstring(
        msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    # print(img.shape)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    x,y,w,h = roi
    # print(w,h)
    img = dst[y:y+h, x:x+w, :]
    (l,w, c) = img.shape
    
    img = img[int(0.1*l):int(0.9*l),int(0.1*w):int(0.8*w), :]
    # print(img.shape)
    image_raw = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(img.shape)
    # cv2.imshow("img",img)    
    (l,w) = img.shape
    s_w = w 
    s_l = l 
    img = cv2.resize(img, (s_w, s_l))
    # print(w, l)
    flow = tracker.track(img)
    # print(flow.shape)
    # density = getDensity(flow)
    # density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)
    # cv2.imshow("density", density)
    mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
    mag = (mag/1.5 * 255).astype('uint8')
    # print(flow.max())
    image_mag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
    image_mag = cv2.resize(image_mag, (w*10, l*10))
    
    # print(image_raw.shape, image_mag.shape, image_arrow.shape)
    # cv2.imshow("mag", mag)
    # img = img[20:-20, 20:-20].copy()
    #cv2.imshow("image", img)
    # cropped_flow = flow[60:-60, 80:-80]
    # flow_sum=np.sum(cropped_flow,axis=1)
    # flow_sum=np.sum(flow_sum,axis=0)
    # if flow_sum_tick:
    #     flow_sum_list.append(flow_sum)
	# print(flow_sum)
    # colors = flow_to_color(cropped_flow)
    #arrows = put_optical_flow_arrows_on_image(
    #np.zeros((cropped_flow.shape[0], cropped_flow.shape[1], 3), dtype="uint8"), cropped_flow)
    # boundary = getBoundary(img)

    # edge = np.zeros((arrows.shape[0], arrows.shape[1]//10, 3), dtype="uint8")
    # summarized_plot = np.hstack((arrows, edge, colors, edge, boundary))
    # cv2.imshow("plot", summarized_plot)
    # cv2.imwrite("/home/yipai/flows/" +
    #             str(rospy.Time.now()) + ".jpg", summarized_plot)
    #cv2.imshow("arrows", arrows)
    # cv2.imshow("colors", colors)
    # cv2.imshow("boundary", boundary)
    # cv2.waitKey(1)
    frame_counter += 1
    if rospy.Time.now() - tick_time > rospy.Duration(secs=1.0):
        # print(img.shape)
        freq = frame_counter / (rospy.Time.now()-tick_time).secs
        frame_counter = 0
        rospy.loginfo("frame rate: " + str(freq))
        mag = np.hypot(flow[:, :, 0], flow[:, :, 1]).max()
        rospy.loginfo("mag: " + str(mag))
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
        # global flow_sum_list
        # global flow_sum_tick
        pass
        # flow_sum_list=[]
        # flow_sum_tick=True
        # print('Start recording!')
        # global flow_sum
        # global flow_sum_list
        # flow_sum_list.append(flow_sum)
        # print(flow_sum)
    elif key == keyboard.Key.down:
        global n
        global image_raw, image_arrow, image_mag, flow, pre_flow, f
        w = 214
        l = 182
        f = cv2.resize(flow, (w*10, l*10))
        if n == 0:
            print(n)
            image_arrow = put_optical_flow_arrows_on_image(cv2.resize(image_raw, (w*10, l*10)), f)
        
        else:
            print(n)
            image_arrow1 = put_optical_flow_arrows_on_image2(cv2.resize(image_raw, (w*10, l*10)), pre_flow)
            image_arrow = put_optical_flow_arrows_on_image(cv2.resize(image_arrow1, (w*10, l*10)), f)
        
        
        pre_flow = f
        
        cv2.imwrite("/home/zgl/Desktop/result/1022/3/image_raw_" + str(n) + ".jpg", cv2.resize(image_raw, (image_arrow.shape[1], image_arrow.shape[0])))
        cv2.imwrite("/home/zgl/Desktop/result/1022/3/image_arrow_" + str(n) + ".jpg", image_arrow)
        cv2.imwrite("/home/zgl/Desktop/result/1022/3/image_flow_" + str(n) + ".jpg", image_mag)
        print("saved")
        n = n + 1
    #     #global flow_sum_tick
        # global flow_sum_list

    #     #flow_sum_tick=False
        # rarr=np.array(flow_sum_list)
        # np.savetxt('test_12.csv',rarr.T,delimiter=',')
        # rospy.loginfo("Flow saved")
    #     # del flow_sum_list[:]


        


def listener():

    rospy.Subscriber("raspicam_node/image/compressed",
                     CompressedImage, Imagecallback, queue_size=1, buff_size=2**22)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()



if __name__ == '__main__':
    listener()
