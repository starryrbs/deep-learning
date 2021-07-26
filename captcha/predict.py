import io

import torch
from torch.autograd import Variable
from PIL import Image

from model import CaptchaNet
from utils import request_captcha_image, label_to_code
from data import transform, get_predict_data_loader

cnn = CaptchaNet()


def predict():
    cnn.eval()
    cnn.load_state_dict(torch.load("./model-90.9.pkl"))
    response = request_captcha_image()
    image = Image.open(io.BytesIO(response.content))
    image.show()
    image = transform(image)
    image = image.unsqueeze(1)  # 加上批量维度
    predict_label = cnn(Variable(image))
    res = label_to_code(predict_label)
    print(f"预测结果：{res}")


if __name__ == '__main__':
    predict()
