import numpy as np

arr1 = np.array([[2,1,2,3],[4,1,1,6],[1,1,1,1],[2,1,1,6],[2,1,2,3],[4,1,1,6],[1,1,1,1],[2,1,1,6]])
height = arr1.shape[0]
width = arr1.shape[1]
arr1[arr1 > 1 ] = 0
arr2 = np.array([[2,1,2,3],[4,1,1,6],[1,1,1,1],[2,1,1,6],[2,1,2,3],[4,1,1,6],[1,1,1,1],[2,1,1,6]])
arr2[arr2 > 1] = 0
mianji = sum(sum(arr2))


for i in range(1,height-1):
    for j in range(1,width-1):
        if(arr1[i-1][j] != 0 and arr1[i][j-1] != 0 and arr1[i][j+1] != 0 and arr1[i+1][j] != 0):
            arr2[i][j] = 0

arr1 = arr2
zhouchang = sum(sum(arr1))


[x,y]=np.where(arr1==True)
print(type(x),y)
print(int(max(x)+min(x))/2)
print(int(max(y)+min(y))/2)
print(4/3.1415*mianji/(zhouchang*zhouchang))
