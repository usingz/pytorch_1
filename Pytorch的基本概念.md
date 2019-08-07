Pytorch是一个深度学习框架，Pytorch的优势在于其编写是动态的，并且速度也算的上比较快。使用Pytorch编写的代码更具有可读性，定义网络结构更为简单并且可以方便的修改某些层，增加某些层等等，方便我们可以按照自己的想法进行“魔改”。
1. 安装Pytorch
   由于我本身是使用Windows系统的（有时候也会用Ubuntu）所以介绍如何在Windows下在安装Pytorch。首先我们将anaconda3安装上，下载地址为[https://www.anaconda.com/download/](https://www.anaconda.com/download/) ，可根据系统类型选择安装。安装好anaconda之后在命令符中输入conda -V，能正确输出anaconda版本号则成功。

  1.1图形化安装Pytorch
  安装好anaconda之后我选择了一种较为简单的方式安装Pytorch，即在anaconda创建的虚拟环境中搜索torch，之后一键点击安装即可。
安装好之后使用一段简单的代码测试一下。
```
# CUDA TEST
import torch
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)
# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))
```
能正常输出结果则安装成功。

  1.2命令行安装Pytorch
  我们打开安装好的anaconda powershell promt，输入命令conda install pytorch即可。这也是较为简单的方法，如出现问题欢迎交流。

2. Pytorch的基础概念
  2.1张量
  Pytorch中的基本单位也是张量英文中叫Tensor，是Pytorch中的基础运算单位，张量表示的是一个多维的矩阵。Pytorch中的张量可以在GPU上运行，而这可以加快我们训练神经网络的时间。下面我们编写一个生成张量的简单例子。
```
import torch
x = torch.rand(2,2)
```
上述代码将会生成一个2行2列的举证，里面的值为随机生成的数值。

  2.2自动求导机制
  Pytorch中的Autograd模块实现了反向传播算法，Autograd可以为所有张量自动提供微分，我们来看一个下例子。
```
import torch
x = torch.rand(2,2,requires_grad=True)
y = torch.rand(2,2,requires_grad=True)
z = torch.sum(x+y)
z.backward()
print(x.grad,y.grad)
```
首先我们创建两个张量，之后对它们进行求和，设置requires_grad是为了对该张量进行自动求导，Pytorch会记录该张量的每一步操作历史并自动计算。之后调用backward()进行求导，计算出结果之后将会保存到对应变良的grad属性里面，我们可以调用x.grad查看梯度。

  3. 使用Pytorch编写神经网络算法
  本次我们选择的是编写手写数字识别的神经网络。由于本人偏向于自然语言处理对图像处理还是了解不深。来看看代码吧。
```
import torch
import torchvision
from torchvision import datasets, transforms
'''
torchvision库是独立pytorch的关于图像操作的工具库，
vision.datasets包括了几个常用的数据集，%%javascript以下载和加载
vision.transforms常用的图像操作，如切割，旋转等
'''
# 加载MNIST手写数字数据集和标签
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,),(0.5,))]
)# 将其归一化


# 导入数据中
trainset = datasets.MNIST(root='data/',train=True,
                         download=True,transform=transform)
trainsetLoader = torch.utils.data.DataLoader(trainset,batch_size=20000, shuffle=True)
testset = datasets.MNIST(root='data/',train=True,
                        download=True,transform=transform)
testsetLoader = torch.utils.data.DataLoader(testset,batch_size=20000,shuffle=True)
# 获取随机数据
dataiter = iter(trainsetLoader)
images, labels = dataiter.next()
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(images[0].numpy().squeeze())
plt.show()
print(images.shape)
print(labels.shape)
# 设计自己的神经网络
import torch.nn as nn
first_in, first_out,second_out = 28*28, 128, 10
model = nn.Sequential(
    nn.Linear(first_in,first_out),# nn.Linear线性层，first_in为输入节点，first_out为输出节点，bias默认为真，线性层会生成一个[out, in]大小的权重矩阵
    nn.ReLU(),# 非线性函数进行激活
    nn.Linear(first_out,second_out), # 我们要做的是十分类
)


# 设置损失函数
loss = nn.CrossEntropyLoss()


# 设置优化函数
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# 开始训练神经网络
for t in range(5):
    for i, batch in enumerate(trainsetLoader,0):
        data,label = batch
        data[0].view(1,784) # 类似于numpy中的reshape，将图片变为一维
        data = data.view(data.shape[0], -1)
        
        # 计算输出
        model_output = model(data)
        #计算损失
        loss_data = loss(model_output, label)
        if i % 200 == 0:
            print('loss is {:.4f}'.format(loss_data))
        # 每次训练之前对梯度清0
        optimizer.zero_grad()
        # 根据误差计算梯度
        loss_data.backward()
        # 迭代优化
        optimizer.step()
    
# 保存下模型
torch.save(model,'data/my_model_recognize_dight.pt')


#将测试数据喂入，查看输出结果
testdataiter =iter(testsetLoader)
testimages, testlabels = testdataiter.next()


ima_vector = testimages[0].squeeze().view(1,-1)
# 模型返回一个1x10的矩阵，
result_digit = model(ima_vector)
print("该图片结果为{}".format(int(result_digit.max(1)[1])),"标签为{}".format(testlabels[0]))
```
至此，我们的一个Pytoch入门练习就到此结束了。
