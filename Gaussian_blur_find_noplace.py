from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from matplotlib import pyplot as plt
import math
from skimage import measure

 
class MyGaussianBlur():
    #初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius=radius
        self.sigema=sigema    
    #高斯的计算公式
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
    #得到滤波模版
    def template(self):
        sideLength=self.radius*2+1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i,j]=self.calc(i-self.radius, j-self.radius)
        all=result.sum()
        return result/all    
    #滤波函数
    def filter(self, image, template): 
        arr=np.array(image)
        height=arr.shape[0]
        width=arr.shape[1]
        newData=np.zeros((height, width))
        for i in range(self.radius, height-self.radius):
            for j in range(self.radius, width-self.radius):
                t=arr[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1]
                a= np.multiply(t, template)
                newData[i, j] = a.sum()
        newImage = Image.fromarray(newData)          
        return newImage
 
r=5 #模版半径，自己自由调整
s=2 #sigema数值，自己自由调整
GBlur=MyGaussianBlur(radius=r, sigema=s)#声明高斯模糊类
temp=GBlur.template()#得到滤波模版



def erzhihua(im1):
    arr1 = np.array(im1)
    height = arr1.shape[0]
    width = arr1.shape[1]
    B = sum(sum(arr1))/height/width
    sig = np.std(arr1 - B)

    image1 = GBlur.filter(im1, temp)#高斯模糊滤波，得到新的图片
    image1 = image1.convert('L')


    threshold = B + sig*sig*0.125
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    image1 = image1.point(table,'1')
    return image1

def find_x_y(arr1):
    label_image = measure.label(arr1)
    label_image2 = measure.label(arr1)
    label = np.max(label_image)
    # np.savetxt('new.csv', label_image, delimiter = ',')

    X = []
    Y = []
    for i in range(1,label + 1): 
        label_image = measure.label(arr1)
        label_image2 = measure.label(arr1)
        height = label_image.shape[0]
        width = label_image.shape[1]
        label_image[label_image > i ] = 0
        label_image[label_image < i ] = 0
        label_image[label_image == i ] = 1
        
        label_image2[label_image2 > i ] = 0
        label_image2[label_image2 < i ] = 0
        label_image2[label_image2 == i ] = 1
        
        mianji = sum(sum(label_image2)) 
        print(mianji)


        for i in range(1,height-1):
            for j in range(1,width-1):
                if(label_image[i-1][j] != 0 and label_image[i][j-1] != 0 and label_image[i][j+1] != 0 and label_image[i+1][j] != 0):
                    label_image2[i][j] = 0

        label_image = label_image2
        zhouchang = sum(sum(label_image)) 


        [x,y]=np.where(label_image==True)
        if x is not None:
            print(int(max(x)+min(x))/2)
            print(int(max(y)+min(y))/2)
        # if(x.size() != 0 and y.size() != 0):
        #     si = 4/3.1415*mianji/(zhouchang*zhouchang)
        #     if(0 < si < 100):
        #         X.append(int(max(x)+min(x))/2)
        #         Y.append(int(max(y)+min(y))/2)
    return X,Y



if __name__ == '__main__':
    im1 =Image.open('D:\python\AI\AI_star_erupt\\3.jpg')
    image1 = erzhihua(im1)
    image1.show()
    arr1 = np.array(image1)
    X,Y = find_x_y(arr1)
    print(X,Y)
    

