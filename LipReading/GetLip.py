import numpy as np
import argparse
import dlib
import cv2
import os
import skvideo.io
import pickle

"""
第一步，设置输入参数
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,
                help="输入单个视频文件路径")
ap.add_argument("-o","--output",required=True,
                help="输出单个视频文件路径")
ap.add_argument("-f","--fps",type=int,default=30,
                help="输出视频的帧率")
ap.add_argument("-c","--codec",type=str,default="MJPG",
                help="输出视频的编码译码器")
args=vars(ap.parse_args())


"""
第二步
    1 提取视频中的每一帧用来处理
    2 从帧中取得唇部
"""
#加载一个预测器路径，不加载也行，可以使用默认的，这个可以换成自己的模型
predictor_path = 'models\\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#切割一个视频后嘴唇要保存到的位置
mouth_destination_path = os.path.dirname(args["output"]) + '/' + 'mouth'
if not os.path.exists(mouth_destination_path):
    os.makedirs(mouth_destination_path)

inputparameters = {}
outputparameters = {}
reader = skvideo.io.FFmpegReader(args["input"],
                                 inputdict=inputparameters,
                                 outputdict=outputparameters)
#得到视频的图像shape
video_shape = reader.getShape()
(num_frames, h, w, c) = video_shape
print(num_frames, h, w, c)

#需要的参数
#可以提取到嘴唇就设置为1，不能提取出嘴唇就设置为0
activation = []
#这个用来计数最大能截取的唇图片数
max_counter = 150
#这个用来获取视频的最大帧数
total_num_frames = int(video_shape[0])
counter = 0
#确定最大取得帧数
num_frames = min(total_num_frames,max_counter)
#cv中正常大小无衬线字体
font = cv2.FONT_HERSHEY_COMPLEX

#用来确定视频输出的writer
writer = skvideo.io.FFmpegWriter(args["output"])

#用于唇部提取的参数
width_crop_max = 0
height_crop_max = 0

"""
第三步
"""
#对于每一帧进行循环处理
for frame in reader.nextFrame():
    print('frame_shape:',frame.shape)

    #如果计数值大于最大帧数，结束
    if counter > num_frames:
        break

    #检测每一个帧,返回多个脸的左上，右下坐标
    detections = detector(frame,1)
    #提取唇部的最后20个特征点
    marks = np.zeros((2,20))
    #所有未归一化的脸部特征

    #如果检测到脸部，输出检测到脸部的个数
    print(len(detections))
    if len(detections) > 0:
        #k是表示第几张脸，d表示对应脸所在的左上和右下位置
        for k, d in enumerate(detections):
            #脸部的shape，返回68个关键点
            shape = predictor(frame, d)
            #co一个中间结点计数值，用来计数当前选了几个和嘴唇相关点
            co = 0
            #找出嘴唇
            for ii in range(48,68):
                """
                循环用于提取和嘴唇相关的特征
                """
                #获取对应的特征点
                X = shape.part(ii)
                A = (X.x, X.y)
                marks[0, co] = X.x
                marks[1, co] = X.y
                co += 1

            #获取截取嘴唇框图的极限值（top-left & bottom-right）
            #X_left表示图像左边，Y_left表示图像下部，X_right表示图像右端，Y_right表示图像上端
            X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                int(np.amax(marks, axis=1)[0]), int(np.amax(marks, axis=1)[1])]

            #得到嘴唇的中心
            X_center = (X_left + X_right) / 2.0
            Y_center = (Y_left + Y_right) / 2.0

            #为裁剪设置一个边界
            #border是设置的边界值
            border = 30
            X_left_new = X_left - border
            Y_left_new = Y_left -border
            X_right_new = X_right + border
            Y_right_new = Y_right + border

            #用于裁剪的框图长宽
            width_new = X_right_new - X_left_new
            height_new = Y_right_new - Y_left_new
            width_current = X_right - X_left
            height_current = Y_right - Y_left

            #确定裁剪矩阵维度（主要任务是生成自适应的区域）
            """
            if width_crop_max == 0 and height_crop_max ==0:
                width_crop_max = width_new
                height_crop_max = height_new
            """

            #得到裁剪框的大小
            X_left_crop = X_left_new
            X_right_crop = X_right_new
            Y_left_crop = Y_left_new
            Y_right_crop = Y_right_new

            if X_left_new >=0 and Y_left_crop >=0 and  X_right_crop < w and Y_right_crop < h:
                #保存嘴部区域
                #注意cv中的图像是先算高度，再算宽度
                mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]
                #mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(mouth_destination_path + '/' + 'frame' + '_' + str(counter) + '.png', mouth)
                print("检测到可裁剪的嘴唇")
                activation.append(1)
            else:
                cv2.putText(frame, "未检测到完整的嘴唇", (30, 30), font, 1, (0, 255, 255), 2)
                print("未检测到完整的嘴唇")
                activation(0)

    else:
        cv2.putText(frame, '没检测到嘴唇. ', (30, 30), font, 1, (0, 0, 255), 2)
        print("没检测到嘴唇")
        activation.append(0)

    #如果检测到嘴唇，在视频中框出嘴唇
    if activation[counter] == 1:
        cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)

    #输出图像
    writer.writeFrame(frame)
    counter +=1

writer.close()

"""
第四步：存储activation向量作为一个列表
python读取列表的方式
    with open(the_filename, 'rb') as f:
        my_list = pickle.load(f)
"""

the_filename = os.path.dirname(args['output'] + '/' + 'activation')
my_list = activation
with open(the_filename, 'wb') as f:
    pickle.dump(my_list, f)