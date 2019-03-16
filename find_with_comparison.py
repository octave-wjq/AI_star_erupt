from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from pylab import *
import math
import numpy as np

im1 =Image.open('D:\python\AI\\3.jpg')
im2 =Image.open('D:\python\AI\\2.jpg')
    
im1 = im1.convert('L')
im2 = im2.convert('L')


arr1 = np.array(im1)
arr2 = np.array(im2)

print(arr1)
print(arr2)

# print(len(arr1),len(arr1[0]))
# print(len(arr2),len(arr2[0]))
arr3 = arr1 - arr2
print(arr3)

arr3 = Image.fromarray(arr3)

# sigma = 150

# height = arr1.shape[0]
# width = arr2.shape[1]
# for i in range(height):
#     for j in range(width):
#         if abs(arr1[i][j] - arr2[i][j])< sigma:
#             arr3[i][j] = 0
#         else:
#             arr3[i][j] = arr1[i][j] - arr2[i][j]
# print(arr1.shape[0])
# print(arr1.shape[1])

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