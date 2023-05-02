import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely import geometry
import math

# Get the directory path of the current Python file
FILE_PATH = os.path.abspath(__file__) # C:/Users/.../Ultra-Fast-Lane-Detection/main.py
DRIVE_NAME, _ = os.path.splitdrive(FILE_PATH) # C:
BASE_DIR = os.path.dirname(FILE_PATH) # C:/Users/.../Ultra-Fast-Lane-Detection

IMG_DIR = os.path.join(BASE_DIR, 'content', 'img', 'road.jpg')
UFLD_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'ufld', 'result.txt')
YOLO_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'yolo', 'result.txt')

def dot_between_two_dot(point_1, point_2):
    new_x = int((point_1[0] + point_2[0]) / 2)
    new_y = int((point_1[1] + point_2[1]) / 2)
    return new_x, new_y

def dis_between_two_dot(point_1, point_2):
    return math.sqrt(((point_1[0] - point_2[0]) ** 2)+((point_1[1] - point_2[1]) ** 2))


def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

def bottom_middle_dot_of_the_square(index):
    x = int(index[2] - ((index[2] - index[0]) / 2))
    y = 720 - index[3]
    return (x, y)

with open(UFLD_RESULT_DIR, 'r') as f:
    file_dir = BASE_DIR + "\\" + f.readline().strip()
    #------------------------------------------------------------------------------#
    file_dir_txt = file_dir + '.txt'
    f2 = open(file_dir_txt, 'r')
    coordination = f2.readlines()   
    
    shape = (720, 1280, 3) # y, x, RGB
    #img = np.zeros(shape, np.uint8)  #純黑色照片
    
    # file_dir_img = file_dir + '.jpg'
    img = cv2.imread(IMG_DIR)

    #------------------------------------------------------------------------------#
    #計算跳行, 每當車道線條跳行時做計算
    jump_count = 0
    slef_line_1 = []  #駕駛車道左邊
    slef_line_2 = []  #駕駛車道右邊

    #------------------------------------------------------------------------------#  車道顯示, 跳線計算

    for i in range(0, len(coordination)):
            index = list(map(int, coordination[i].split()))    
            if(list(map(int, coordination[i-1].split()))[1] < list(map(int, coordination[i].split()))[1]):
                jump_count += 1
            if(jump_count == 2 or jump_count == 3):
                cv2.circle(img,index,5,(255,0,0),-1)
                if(jump_count == 2):
                    slef_line_1.append(index)
                else:
                    slef_line_2.append(index)  
            else:
                cv2.circle(img,index,5,(0,255,0),-1)
                index = list(map(int, coordination[i].split()))

    f.close()
    
    #------------------------------------------------------------------------------# 行車方向標記
    num = min(len(slef_line_1), len(slef_line_2))#中點數量取最小值
    lane_middle_dot = []                         #中點座標
    for i in range(num):
        coor = []
        x, y = dot_between_two_dot(slef_line_1[i], slef_line_2[i])
        coor = [x, y]
        lane_middle_dot.append(coor)
        pad = 20
        #if(i % 3 == 0):
        if(i):
            cv2.circle(img,coor,5,(0,0,255),-1)
    #------------------------------------------------#箭頭標記      

    if(i == (num-7)):
        cv2.circle(img,(coor[0]+pad, coor[1]),5,(0,0,255),-1)
        cv2.circle(img,(coor[0]-pad, coor[1]),5,(0,0,255),-1)
    elif(i == (num-9)):
        cv2.circle(img,(coor[0]+pad*2, coor[1]),5,(0,0,255),-1)
        cv2.circle(img,(coor[0]-pad*2, coor[1]),5,(0,0,255),-1)
    elif(i == (num-11)):
        cv2.circle(img,(coor[0]+pad*3, coor[1]),5,(0,0,255),-1)
        cv2.circle(img,(coor[0]-pad*3, coor[1]),5,(0,0,255),-1)


    #------------------------------------------------------------------------------#  車框顯示
    f2 = open(YOLO_RESULT_DIR, 'r')
    file_dir_txt = BASE_DIR + "\\" + f2.readline().strip() + '.txt'
    #------------------------------------------------------------------------------#
    f3 = open(file_dir_txt, 'r')
    car_coordination = f3.readlines()        

#for i in range(0, len(car_coordination)):
    #index = list(map(int, map(float, car_coordination[i].split())))   
    #img = cv2.rectangle(img, (index[0], index[1]), (index[2], index[3]), (255, 0, 0), 2)


#-----------------------------------------------------------------#

    # count = 0
    # square = [(slef_line_1[44][0],0), (slef_line_2[44][0],0), (slef_line_2[44][0],720-(slef_line_2[44][1])), (slef_line_1[44][0],720-(slef_line_1[44][1]))] 

    # for i in range(0, len(car_coordination)):
    #     index = list(map(int, map(float, car_coordination[i].split())))   
    #     coor = bottom_middle_dot_of_the_square(index)                              
    #     if(if_inPoly(square, coor)):
    #         img = cv2.rectangle(img, (index[0], index[1]), (index[2], index[3]), (0,0,255), 2)
        

    f2.close()
    f3.close()

#------------------------------------------------------------------------------#利用車道中點計算
#取離中點最近的距離作為閥值 ex:lane_middle_dot[3]為最小 為10.04987562112089
    small_value = 1000000
    for i in range(0, len(car_coordination)):
        box_index = list(map(int, map(float, car_coordination[i].split())))   
        coor = bottom_middle_dot_of_the_square(box_index)   
        for j in range(0, len(lane_middle_dot)):
            if(dis_between_two_dot(coor, lane_middle_dot[j]) <= small_value):
                small_value = dis_between_two_dot(coor, lane_middle_dot[j])



    #加15如果不行就是使用兩車道中點距離的一半做閥值
    for i in range(0, len(car_coordination)):
        box_index = list(map(int, map(float, car_coordination[i].split())))   
        coor = bottom_middle_dot_of_the_square(box_index)   
        for j in range(0, len(lane_middle_dot)):
            if(dis_between_two_dot(coor, lane_middle_dot[j]) <= small_value + 15):
                img = cv2.rectangle(img, (box_index[0], box_index[1]), (box_index[2], box_index[3]), (0,0,255), 2)

                

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()