#导入numpy和matplotlib
#numpy提供对矩阵丰富的计算
#matplotlib根据数据进行绘图
import numpy as np
from matplotlib import pyplot as plt

#数据集
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

#定义前向传播函数
def foward(x,w):
    return x*w

#定义损失函数
def loss(x,y,w):
    y_pred=foward(x,w)
    return (y_pred-y)*(y_pred-y)

#w值的列表
w_list=[]
#均方差值的列表
mse_list=[]

#w从0.0-4.1，跨度为0.1
for w in np.arange(0.0,4.1,0.1):
    print('w=',w)
    #l_sum:损失值之和
    l_sum = 0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=foward(x_val,w)
        loss_val=loss(x_val,y_val,w)
        l_sum+=loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print('MSE=',l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

#根据均方差和w绘制图像
plt.title('Linear Model')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.plot(w_list,mse_list)
plt.show()
