import csv
import cv2
import os

#训练集文件路径
filepath="D:\\Files\\af2019-cv-training-20190312"
#图片保存的路径---原图
targetpath1="static_img_20"
#图片保存的路径---顺时针90度旋转
targetpath2="static_img_20_r"
#图片保存的路径---逆时针90度旋转
targetpath3="static_img_20_l"


#裁剪的区域半边长
offset=10
#裁剪的图片数量
size=6289
#裁剪的图片类型 静态图 新图 原图
graph = ['a','b','c']


csv_file = csv.reader(open("filepath\\list.csv"))
img_name = [row[0] for row in csv_file]
csv_file = csv.reader(open("filepath\\list.csv"))
x_pos = [row[1] for row in csv_file]
csv_file = csv.reader(open("filepath\\list.csv"))
y_pos = [row[2] for row in csv_file]
csv_file = csv.reader(open("filepath\\list.csv"))
judge = [row[3] for row in csv_file]

#顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img
#逆时针旋转90度
def RotateAntiClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip( trans_img, 0 )
    return new_img

#创建文件夹
def makedirs(targetpath):

    if not os.path.exists(targetpath):
        os.mkdir(targetpath)
    if not os.path.exists(targetpath+"\\newtarget"):
            os.mkdir(targetpath+"\\newtarget")
    if not os.path.exists(targetpath+"\\isstar"):
            os.mkdir(targetpath+"\\isstar")
    if not os.path.exists(targetpath+"\\known"):
            os.mkdir(targetpath+"\\known")
    if not os.path.exists(targetpath+"\\ghost"):
            os.mkdir(targetpath+"\\ghost")
    if not os.path.exists(targetpath+"\\pity"):
            os.mkdir(targetpath+"\\pity")
    if not os.path.exists(targetpath+"\\noise"):
            os.mkdir(targetpath+"\\noise")
    if not os.path.exists(targetpath+"\\asteroid"):
            os.mkdir(targetpath+"\\asteroid")
    if not os.path.exists(targetpath+"\\isnova"):
            os.mkdir(targetpath+"\\isnova")

#写入文件
def writefiles(targetpath,cropped):
    if judge[i] == "newtarget":
        cv2.imwrite(targetpath+"\\newtarget\\"+img_name[i]+".jpg", cropped) 
    elif judge[i] == "isstar":
        
        cv2.imwrite(targetpath+"\\isstar\\"+img_name[i]+".jpg", cropped)
    elif judge[i] == "known":
        
        cv2.imwrite(targetpath+"\\known\\"+img_name[i]+".jpg", cropped)
    elif judge[i] == "ghost":
        
        cv2.imwrite(targetpath+"\\ghost\\"+img_name[i]+".jpg", cropped)
    elif judge[i] == "pity":
        
        cv2.imwrite(targetpath+"\\pity\\"+img_name[i]+".jpg", cropped)
    elif judge[i] == "noise":
        
        cv2.imwrite(targetpath+"\\noise\\"+img_name[i]+".jpg", cropped)
    elif judge[i] == "asteroid":
        
        cv2.imwrite(targetpath+"\\asteroid\\"+img_name[i]+".jpg", cropped)
    elif judge[i] == "isnova":
        
        cv2.imwrite(targetpath+"\\isnova\\"+img_name[i]+".jpg", cropped)



for i in range(1,size):
    img = cv2.imread(filepath+"\\"+img_name[i][0:2]+"\\"+img_name[i]+"_"+graph[0]+".jpg")
    xlabel_l = int(x_pos[i])-offset
    xlabel_r = int(x_pos[i])+offset
    ylabel_l = int(y_pos[i])-offset
    ylabel_r = int(y_pos[i])+offset
    if xlabel_l<0:
        xlabel_l = 0
    if xlabel_r>img.shape[1]:
        xlabel_r = img.shape[1]
    if ylabel_l<0:
        ylabel_l = 0
    if ylabel_r>img.shape[0]:
        ylabel_r = img.shape[0]
    print(img.shape)
    print(xlabel_l,xlabel_r,ylabel_l,ylabel_r)
    cropped = img[ylabel_l:ylabel_r,xlabel_l:xlabel_r]  # 裁剪坐标为[y0:y1, x0:x1]
    rotatecropped_r = RotateClockWise90(cropped)
    rotatecropped_l = RotateAntiClockWise90(cropped)

    makedirs(targetpath1)
    writefiles(targetpath1,cropped)
    makedirs(targetpath2)
    writefiles(targetpath2,rotatecropped_r)
    makedirs(targetpath3)
    writefiles(targetpath3,rotatecropped_l)
    

