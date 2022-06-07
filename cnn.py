from torch import nn

class CNN(nn.Module):#定义卷积神经网络

    def __init__(self):

        super(CNN, self).__init__()

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



        self.fc = nn.Sequential(

            nn.Linear(50 * 5 * 5, 1024),

            nn.ReLU(inplace=True),

            nn.Linear(1024, 128),

            nn.ReLU(inplace=True),

            nn.Linear(128, 10)

        )




    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
