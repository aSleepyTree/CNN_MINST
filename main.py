# 导入包，在测试时写了很多东西，于是莫名其妙的导入了很多没用到包没有删除
import torch

import torchvision.transforms as transforms

from PIL import Image

from torch import nn, optim

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import datasets

import cv2

import numpy as np

import cnn

clean = 0
board = 200
WINDOWNAME = 'Win'
line = 10
#鼠标划线函数
def draw_line(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif (event == cv2.EVENT_MOUSEMOVE) & (flags == cv2.EVENT_FLAG_LBUTTON):
        cv2.line(img, (ix, iy), (x, y), 255, line)#黑线，粗细为line像素
        ix, iy = x, y


img = np.zeros((board, board, 1), np.uint8)
cv2.namedWindow(WINDOWNAME)
cv2.setMouseCallback(WINDOWNAME, draw_line)
transf = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize([0.5], [0.5])
     ])
while True:
        cv2.imshow(WINDOWNAME, img)#展示写入窗口，输入就是使用鼠标在其上写数字从0到9，写完后按下空格进行识别，再按一下清空窗口，写好后再按再次检测，往复；按esc退出
        key = cv2.waitKey(20)

        if key == 32:
            if clean:#重新初始化画板
                img = np.zeros((board, board, 1), np.uint8)
                cv2.imshow(WINDOWNAME, img)
            else:
                # 把图片resize成MNIST数据集的标准尺寸28*28
                resized_img = cv2.resize(img, (28, 28), cv2.INTER_CUBIC)
                im = transf(resized_img)
                im = torch.unsqueeze(im, dim=0)  # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]
                if torch.cuda.is_available():#GPU是否可以使用
                    im = im.cuda()
                else:
                    im = Variable(im)
                model  = torch.load('CNN_for_MNIST.pth')
                out = model(im)
                c = out.tolist()[0]
                print(c)
                print(c.index(max(c)))
            clean = not clean
        elif key == 27:
            break
cv2.destroyAllWindows()




