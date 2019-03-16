from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import time

import math
import numpy as np

 
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
# im=Image.open('D:\python\AI\\test1.jpg')#打开图片

# image.show()#图片显示
im1 =Image.open('D:\python\AI\\1.jpg')
# im2 =Image.open('D:\python\AI\\2.jpg')

image1 = GBlur.filter(im1, temp)#高斯模糊滤波，得到新的图片
# image2 = GBlur.filter(im2,temp)
image1 = image1.convert('L')
threshold = 120
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)

image1 = image1.point(table,'1')
# image2 = image2.point(table,'1')

image1.show()
# image2.show()


# arr1 = np.array(image1)
# arr2 = np.array(image2)
# arr3 = abs(arr2 - arr1)
# print(np.where(arr3 == np.max(arr3)))
# height = arr1.shape[0]
# width = arr2.shape[1]
# maxheigt = 0
# maxLight =0
# max = 0
# for i in range(arr1):
#     for j in range(arr2):
#         if abs(arr1[i][j] - arr2[i][j]
# image.show()

# image.show()