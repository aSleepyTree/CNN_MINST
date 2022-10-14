# CNN_MINST
CNN MINST 手写数字识别 中科大自动化系人工智能导论2022大作业三

以下引自作业报告
<div align='center' ><font size='6'>基于MNIST数据集的手写数字识别 </font></div>

# 实验环境

----

python 3.10.2

pyTorch 1.11.0

pillow 9.1.0

opencv 4.55.64

numpy 1.22.3

# 摘要

----

使用pyTorch定义了一个两层的卷积神经网络，使用MNIST数据集对其进行训练，使用训练后的模型进行手写数字的识别，同时提供了一个查看MNIST数据集的方法。文件中已经给出了一个训练好的模型即`CNN_for_MNIST.pth`；`__pycache__`文件夹似乎是由Python自动生成的，目的是使程序运行的稍快一些【<https://stackoverflow.com/questions/16869024/what-is-pycache>】
建议不用关心或者嫌麻烦直接删除掉该文件夹

# 代码结构

----

- `cnn.py`定义一个两层（两个卷积层、池化层和一个全连接层）的卷积神经网络
- `train.py`用于使用cnn定义的网络来训练模型（使用MNIST数据集->[MNIST数据集](http://yann.lecun.com/exdb/mnist/)）
- `tt.py`提供了一种查看MNIST数据集的方式，打开运行之后即可查看训练所用的二进制数据相应的的图片，由于pyplot的显示的窗口相关操作我不太理解，故需要关闭当前显示窗口才能查看下一张图片，不关心训练集的话可以不管这个文件
- `main.py`使用训练好的模型进行手写数字的识别

# 代码细节

----

- `cnn.py`文件定义了一个CNN类，其中的

        self.layer1 = nn.Sequential(#卷积
            nn.Conv2d(1, 25, kernel_size=3),#输入通道，输出通道，卷积核

            nn.BatchNorm2d(25),#参数为输出通道数

            nn.ReLU(inplace=True)#线性整流

        )



        self.layer2 = nn.Sequential(#池化

            nn.MaxPool2d(kernel_size=2, stride=2)

        )



        self.layer3 = nn.Sequential(#卷积

            nn.Conv2d(25, 50, kernel_size=3),

            nn.BatchNorm2d(50),

            nn.ReLU(inplace=True)

        )



        self.layer4 = nn.Sequential(#池化

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

定义了了一个神经网络的两个卷积层和两个池化层

而代码块

        self.fc = nn.Sequential(

            nn.Linear(50 * 5 * 5, 1024),

            nn.ReLU(inplace=True),

            nn.Linear(1024, 128),

            nn.ReLU(inplace=True),

            nn.Linear(128, 10)

        )
将定义了后续的两次线性化和三次激活

- `train.py`文件相应操作在代码中有注释，这里只描述过程

  - 定义数据分批大小即一次训练所用样本数、学习率和数据预处理方法（将图片转换成torch用的tensor格式）；选择模型之后的损失函数和优化器定义也属于此块（torch中已有模型，这里只做调用）
  - 下载数据（文件中已经下载好了放在data文件夹中）
  - 选择模型即cnn中已经定义的、使用数据进行训练、每50次训练输出当前错误率
  - 训练完成后保存模型（由于本人已经训练过模型，这里在提交的代码中已经被注释）
  - 在测试集中看模型识别准确率，代码中将predict和label进行了输出，可以直观的看到识别情况

- `main.py`文件，同上这里描述过程

  - 首先定义了一些参数：画板大小、画线宽度
  - 定义画线函数
  - 循环读取画线进行识别，画好后按`space`进行识别，识别完成后按`space`进行清空可继续进行，任何时候按`esc`可退出

