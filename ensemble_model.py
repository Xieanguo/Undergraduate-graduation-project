import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform)

# 加载模型
model1 = torch.load('resnet50_cifar10.pth')
model2 = torch.load('resnet152_cifar10.pth')
model3 = torch.load('efficientnetb4_cifar10.pth')
model4 = torch.load('efficientnetb7_cifar10.pth')

# 在GPU上进行模型预测
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)

def ensemble_vote(models, weights, data_loader):
    """
    使用投票法对多个模型进行集成学习
    :param models: 待集成的模型列表
    :param weights: 每个模型对应的权重列表
    :param data_loader: 测试数据加载器
    :return: 预测标签
    """
    # 模型评估模式
    for model in models:
        model.eval()

    # 初始化预测结果列表
    predictions = torch.zeros(len(data_loader.dataset))

    # 在GPU上进行预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)

    # 遍历数据并投票
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = torch.zeros(inputs.size(0), 10).to(device)

            # 遍历模型并加权求和预测输出
            for model, weight in zip(models, weights):
                outputs += weight * model(inputs)

            # 投票
            _, predicted = torch.max(outputs, 1)
            predictions[i*inputs.size(0):(i+1)*inputs.size(0)] = predicted

    return predictions

# 创建测试数据加载器
batch_size = 1000
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型权重
weights = [2, 3, 5, 4]

# 使用软投票集成模型
models = [model1, model2, model3, model4]
predictions = ensemble_vote(models, weights, test_loader)

# 计算准确率
targets_tensor = torch.tensor(test_dataset.targets)
correct = (predictions == targets_tensor).sum().item()
total = len(test_dataset)
accuracy = correct / total
print('集成模型的准确率：{:.2f}%'.format(accuracy * 100))

# 将投票法集成学习的模型保存到本地
ensemble_model = models
torch.save(ensemble_model, 'ensemble_model.pth')
