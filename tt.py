import torch
import torchvision
import matplotlib.pyplot as plt #用于显示图片
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#忽略警告
import warnings
warnings.filterwarnings('ignore')

#选择运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#下载数据集    28*28=784
dataset_train = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataset_test = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=False)

#将数据集按批量大小加载到数据集中
data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100, shuffle=True)  #600*100*([[28*28],x])
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=100, shuffle=False)

#for epoch in range(5):  #一共五个周期，其中一个周期（len(dataset_train)=60000）/(batch_size=100)=（len(dataset_train)=600）个批量
for i, (images, labels) in enumerate(data_loader_train):

    #print(i, images[0].shape, labels[0].shape)
    '''
        每一个周期，共600个批次（i=0~599）；
        data_loader_train包含600个批次，包括整个训练集；
        每一批次一共100张图片，对应100个标签, len(images[0])=1；
        images包含一个批次的100张图片（image[0].shape=torch.Size([1,28,28])），labels包含一个批次的100个标签，标签范围为0~9
    '''

    #每20个批量绘制最后一个批量的所有图片
    if (i + 1) % 3 == 0:
        for j in range(len(images)):
            #if(labels[j] == 1):
                print('batch_number [{}/{}]'.format(i + 1, len(data_loader_train)))
                image = images[j].resize(28, 28) #将(1,28,28)->(28,28)
                print(image)
                plt.imshow(image)  # 显示图片,接受tensors, numpy arrays, numbers, dicts or lists
                plt.axis('off')  # 不显示坐标轴
                plt.title("$The {} picture in {} batch, label={}$".format(j + 1, i + 1, labels[j]))
                plt.show()
            
