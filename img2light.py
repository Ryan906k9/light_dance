import paddlehub as hub
import cv2
import numpy as np
from paddlehub.common.logger import logger
import os
import time
import math






f_list = os.listdir('images')
img_num = 0

for i in f_list:
    if '.jpg' in i:
        img_num += 1

f_list_path = []
for f in range(img_num):
    f_list_path.append('images/'+str(f+1)+'.jpg')
print(f_list_path)

module = hub.Module(name="pose_resnet50_mpii")


# set input dict
input_dict = {"image": f_list_path}

# execute predict and print the result
results = module.keypoint_detection(data=input_dict)
'''
for result in results:
    print(result['path'])
    for i in result['data'].items():
        print(i)
    print('\t')
'''
# 加入人脸检测

'''
def get_face_landmark(image):
    """
    预测人脸的68个关键点坐标
    images(ndarray): 单张图片的像素数据
    """
    module = hub.Module(name="face_landmark_localization")
    try:
        # 选择GPU运行，use_gpu=True，并且在运行整个教程代码之前设置CUDA_VISIBLE_DEVICES环境变量
        res = module.keypoint_detection(images=[image], use_gpu=False)
        return True, res[0]['data'][0]
    except Exception as e:
        logger.error("Get face landmark localization failed! Exception: %s " % e)
        return False, None
'''
module = hub.Module(name="face_landmark_localization")

for i in range(img_num):
    img = cv2.imread('images/'+str(i+1)+'.jpg', cv2.IMREAD_COLOR)
    res = module.keypoint_detection(images=[img], use_gpu=False)
    face_landmark = res[0]['data'][0]
    nose_point = np.array([
                face_landmark[31]
            ], dtype='int')
    eyes_points = np.array([
                face_landmark[37],face_landmark[46]
            ], dtype='int')
    head_point = np.array([
                face_landmark[28]
            ], dtype='int')

    # 耳朵下缘点为 3 和 15
    ear_points = np.array([
                face_landmark[3],face_landmark[14]
            ], dtype='int')

    #print(eyes_points)


    img = np.zeros(img.shape, dtype = np.uint8)


    cv2.line(img,tuple(results[i]['data']['left_ankle']),tuple(results[i]['data']['left_knee']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['left_hip']),tuple(results[i]['data']['left_knee']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['right_ankle']),tuple(results[i]['data']['right_knee']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['right_hip']),tuple(results[i]['data']['right_knee']),(96,255,0))

    cv2.line(img,tuple(results[i]['data']['left_shoulder']),tuple(results[i]['data']['left_elbow']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['left_wrist']),tuple(results[i]['data']['left_elbow']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['right_shoulder']),tuple(results[i]['data']['right_elbow']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['right_wrist']),tuple(results[i]['data']['right_elbow']),(96,255,0))

    cv2.line(img,tuple(results[i]['data']['right_hip']),tuple(results[i]['data']['pelvis']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['left_hip']),tuple(results[i]['data']['pelvis']),(96,255,0))

    cv2.line(img,tuple(results[i]['data']['left_shoulder']),tuple(results[i]['data']['thorax']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['right_shoulder']),tuple(results[i]['data']['thorax']),(96,255,0))

    #cv2.line(img,tuple(results[i]['data']['head top']),tuple(results[i]['data']['upper neck']),(96,255,0))
    #cv2.line(img,tuple(results[i]['data']['upper neck']),tuple(results[i]['data']['thorax']),(96,255,0))
    cv2.line(img,tuple(results[i]['data']['thorax']),tuple(results[i]['data']['pelvis']),(96,255,0))

    # 计算肋骨的点坐标
    left_1 = ((results[i]['data']['left_shoulder'][0]+results[i]['data']['left_hip'][0])//2,
              (results[i]['data']['left_shoulder'][1]+results[i]['data']['left_hip'][1])//2)
    right_1 = ((results[i]['data']['right_shoulder'][0]+results[i]['data']['right_hip'][0])//2,
              (results[i]['data']['right_shoulder'][1]+results[i]['data']['right_hip'][1])//2)
    left_2 = ((results[i]['data']['left_shoulder'][0]+left_1[0])//2,
              (results[i]['data']['left_shoulder'][1]+left_1[1])//2)
    right_2 = ((results[i]['data']['right_shoulder'][0]+right_1[0])//2,
              (results[i]['data']['right_shoulder'][1]+right_1[1])//2)
    left_3 = ((results[i]['data']['left_hip'][0]+left_1[0])//2,
              (results[i]['data']['left_hip'][1]+left_1[1])//2)
    right_3 = ((results[i]['data']['right_hip'][0]+right_1[0])//2,
              (results[i]['data']['right_hip'][1]+right_1[1])//2)

    cv2.line(img,left_1,right_1,(96,255,0))
    cv2.line(img,left_2,right_2,(96,255,0))
    cv2.line(img,left_3,right_3,(96,255,0))

    # 在手上画圈
    cv2.circle(img,tuple(results[i]['data']['left_wrist']),10,(96,255,0))
    cv2.circle(img,tuple(results[i]['data']['right_wrist']),10,(96,255,0))



    # 画出眼镜，采用额头的点，稳定性会好些。先画上缘，再画下面的镜框（这里先尝试三角形眼镜）
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]+20),(head_point[0][1])-2]),(96,255,0))
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]-20),(head_point[0][1])-2]),(96,255,0))
    # 计算眼镜下缘坐标，横坐标取边缘点，纵坐标往下取20
    cv2.line(img,tuple([(head_point[0][0]+20),(head_point[0][1])-2]),tuple([(head_point[0][0]+20),(head_point[0][1])+15]),(96,255,0))
    cv2.line(img,tuple([(head_point[0][0]-20),(head_point[0][1])-2]),tuple([(head_point[0][0]-20),(head_point[0][1])+15]),(96,255,0))
    # 连接下缘和额头
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]+20),(head_point[0][1])+15]),(96,255,0))
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]-20),(head_point[0][1])+15]),(96,255,0))

    # 画出鼻子，梯形，上缘空白
    cv2.line(img,tuple([nose_point[0][0]-5,nose_point[0][1]]),tuple([nose_point[0][0]-12,nose_point[0][1]+10]),(96,255,0))
    cv2.line(img,tuple([nose_point[0][0]+5,nose_point[0][1]]),tuple([nose_point[0][0]+12,nose_point[0][1]+10]),(96,255,0))
    cv2.line(img,tuple([nose_point[0][0]-12,nose_point[0][1]+10]),tuple([nose_point[0][0]+12,nose_point[0][1]+10]),(96,255,0))

    # 画出鼻环
    cv2.circle(img,tuple([nose_point[0][0],nose_point[0][1]+20]),10,(96,255,0))

    # 画出额头的羽毛
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]-50),(head_point[0][1]-50)]),(96,255,0))
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]+50),(head_point[0][1]-50)]),(96,255,0))
    cv2.line(img,tuple(head_point[0]),tuple([(head_point[0][0]),(head_point[0][1]-70)]),(96,255,0))

    # 画出耳坠和耳环
    cv2.line(img,tuple(ear_points[0]),tuple([(ear_points[0][0]),(ear_points[0][1]+25)]),(96,255,0))
    cv2.line(img,tuple(ear_points[1]),tuple([(ear_points[1][0]),(ear_points[1][1]+25)]),(96,255,0))
    cv2.circle(img,tuple([(ear_points[0][0]),(ear_points[0][1]+25)]),10,(96,255,0))
    cv2.circle(img,tuple([(ear_points[1][0]),(ear_points[1][1]+25)]),10,(96,255,0))


    cv2.imwrite("images_light/"+str(i+1)+".jpg",img)


