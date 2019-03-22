from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from pylab import *
import math

im1 =Image.open('D:\python\AI\AI_star_erupt\\1.jpg')
im2 =Image.open('D:\python\AI\AI_star_erupt\\2.jpg')
    
im1 = im1.convert('L')
im2 = im2.convert('L')


arr1 = np.array(im1)
arr2 = np.array(im2)

# print(arr1)
# print(arr2)

# print(len(arr1),len(arr1[0]))
# print(len(arr2),len(arr2[0]))
# arr3 = arr1 - arr2
# print(arr3)



sigma = 120
# height = arr1.shape[0]
# width = arr2.shape[1]
# arr3 = np.zeros((height,width))
# for i in range(height):
#     for j in range(width):
#         if (max(arr1[i][j],arr2[i][j]) - min(arr1[i][j],arr2[i][j]))< sigma:
#             arr3[i][j] = 0
#         else:
#             arr3[i][j] = max(arr1[i][j],arr2[i][j]) - min(arr1[i][j],arr2[i][j])

# arr3 = Image.fromarray(arr3)
# arr3.show()

height = arr1.shape[0]
width = arr2.shape[1]
arr3 = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        if arr1[i][j] - arr2[i][j]< sigma:
            arr3[i][j] = 0
        else:
            arr3[i][j] = arr1[i][j] - arr2[i][j]

arr3 = Image.fromarray(arr3)
arr3.show()
# threshold = 200
# table = []
# for i in range(256):
#     if i < threshold:
#         table.append(0)
#     else:
#         table.append(1)

# im1 = im1.point(table,'1')

# arr1 = np.array(im1)

# im1.show()
# print(np.where(arr1 == True))