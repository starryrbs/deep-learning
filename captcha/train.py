import torch
import torch.nn as nn
from torch.autograd import Variable

from captcha.utils import label_to_code
from data import get_test_data_loader, get_train_data_loader
from model import CaptchaNet
from one_hot_encoding import decode

num_epochs = 100
learning_rate = 0.001

cnn = CaptchaNet()


def train():
    # 切换模型到训练模式
    cnn.train()
    print("初始化网络")
    criterion = nn.MultiLabelSoftMarginLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)  # 定义优化器

    train_data_loader = get_train_data_loader()
    print(f"训练集图片：{len(train_data_loader) * 64}")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_loader):
            images = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images)
            # print(f"预测的labels：{label_to_code(predict_labels)},实际的labels:{label_to_code(labels)}")
            loss = criterion(predict_labels, labels)  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
    print("save last model")


def test():
    cnn.eval()
    # cnn.load_state_dict(torch.load("./model.pkl"))

    total = 0
    correct = 0
    test_data_loader = get_test_data_loader()
    with torch.no_grad():
        for images, labels in test_data_loader:
            predict_label = cnn(Variable(images))
            predict_label = label_to_code(predict_label)
            true_label = decode(labels.numpy()[0])
            print(f"预测的labels：{predict_label},实际的labels:{true_label}")
            total += labels.size(0)
            if predict_label == true_label:
                correct += 1
            if total % 200 == 0:
                print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
        print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    train()
    test()
