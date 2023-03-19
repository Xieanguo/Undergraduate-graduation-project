import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm

#调整预训练模型安装目录
os.environ['TORCH_HOME']='F:/graduation_project/graduation/models'

# 超参数设置
batch_size = 500
num_epochs = 80
learning_rate = 0.003

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./cifar10_data', train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./cifar10_data', train=False, download=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model = timm.create_model('tf_efficientnet_b4', pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 10)
model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=7e-5,amsgrad=True)

# 学习率衰减
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# 训练模型
if __name__=='__main__':
    total_step = len(train_loader)
    print("---------------开始训练---------------")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        lr_scheduler.step()  # 执行学习率衰减

        # 测试模型
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # 保存模型
    torch.save(model, 'efficientnetb4_cifar10.pth')
    print("Model saved.")