"""
从指定链接获取验证码图片，并给图片打标签
"""
import io
import os

from PIL import Image
import matplotlib.pyplot as plt

from captcha.utils import request_captcha_image


def get_captcha_image(count=10):
    for index in range(count):
        response = request_captcha_image()
        image = Image.open(io.BytesIO(response.content))

        plt.figure(figsize=(4, 4))
        plt.ion()  # 打开交互模式
        plt.axis('off')  # 不需要坐标轴
        plt.imshow(image)
        plt.pause(0.1)

        code = input(f"第【{index}】个请输入验证码：")
        image_path = os.path.join(os.path.dirname(__file__), "data/predict", f"{index}-{code}.jpg")

        with open(image_path, "wb") as fp:
            fp.write(response.content)

        plt.clf()  # 清空图片
        plt.close()


if __name__ == '__main__':
    get_captcha_image(count=500)
