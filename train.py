#模型训练，作用为下载MNIST数据集(放入同目录下的data文件夹),使用数据集训练网络并将训练好的模型保存为同目录下的CNN_for_MNIST.pth
import torch

from torch import nn, optim

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import cv2

import cnn


# 定义一些超参数

batch_size = 64         #分批参数

learning_rate = 0.02    #学习率




# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）

# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差

# transforms.Compose()函数将各种预处理的操作组合到了一起

data_tf = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize([0.5], [0.5])
     ])


# 数据集的下载器,下载数据集放入data文件夹

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True) #训练集数据

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)                #测试集数据

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 选择模型

model = cnn.CNN()

if torch.cuda.is_available():#关于是否使用GPU，本人未使用，但看网上大家都会写上这个我也加上去了，可能GPU效率更高吧，但就本问题和模型而言计算量CPU就可以胜任，下面出现也类似

    model = model.cuda()


# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()#常用于多分类问题的交叉熵

optimizer = optim.SGD(model.parameters(), lr=learning_rate)




# 训练模型

epoch = 0

#从训练集读取数据，开始迭代

for data in train_loader:

    img, label = data
    
    if torch.cuda.is_available():

        img = img.cuda()
        
        label = label.cuda()

    else:

        img = Variable(img)

        label = Variable(label)

    out = model(img)

    loss = criterion(out, label)    #计算损失函数值

    print_loss = loss.data.item()

    optimizer.zero_grad()   #将梯度归零

    loss.backward()         #反向传播计算得到每个参数的梯度值

    optimizer.step()        #通过梯度下降执行一步参数更新

    epoch+=1
    
    if epoch%50 == 0:       #输出进度

        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))


# 保存和加载整个模型

#torch.save(model, 'CNN_for_MNIST.pth')




# 模型评估

model.eval()

eval_loss = 0

eval_acc = 0

for data in test_loader:

    img, label = data
    img = Variable(img)

    if torch.cuda.is_available():       #CPU or GPU

        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)         #pred为一个tensor,里面是一组预测可能性最大的数,实际为下标，但在这里下标刚好与数匹配
    num_correct = (pred == label).sum() #按理说bool型不能接sum，但当pred和label是tensor格式时会得到一个tensor且只有一个值为前两者相等的值得个数，刚好统计了正确的预测数
    print('pred',pred)
    print('label',label)
    #print(type(num_correct))
    #print(num_correct)
    #print((pred == label).sum())
    
    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(      #输出模型评估参数总损失和正确率

    eval_loss / (len(test_dataset)),

    eval_acc / (len(test_dataset))

))
