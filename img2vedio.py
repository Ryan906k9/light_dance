import cv2
import os

# 查看原始视频的参数
cap = cv2.VideoCapture('22.mp4')
ret, frame = cap.read()
height=frame.shape[0]
width=frame.shape[1]
fps = cap.get(cv2.CAP_PROP_FPS)  #返回视频的fps--帧率
size=cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #返回视频的宽，等同于frame.shape[1]
size1=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  #返回视频的高，等同于frame.shape[0]

#把参数用到我们要创建的视频上
video = cv2.VideoWriter('22_light.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #创建视频流对象

path = '22_images_light/'
file_list = os.listdir(path)
img_num = len(file_list)

for i in range(img_num):
    #if item.endswith('.jpg'):   #判断图片后缀是否是.png
    item = path + str(i+1) + '.png'
    img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
    video.write(img)        #把图片写进视频
video.release() #释放