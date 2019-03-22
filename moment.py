import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import measure

img = cv2.imread('5.jpg',0)
label_image = measure.label(kk)

# ret,thresh = cv2.threshold(img,127,255,0)
# image,contours,hierarchy=cv2.findContours(thresh,1,2)
# cnt=contours[0]
# M=cv2.moments(cnt)
# cx=int(M['m10']/M['m00'])
# cy=int(M['m01']/M['m00'])
# print(M)
# print(cx,cy)

# x,y,w,h=cv2.boundingRect(cnt)
# print(x,y,w,h)
# img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# plt.imshow(img,cmap='gray')
# plt.show()