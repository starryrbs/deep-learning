import numpy as np
import requests

ALL_CHARSET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

captcha_image_url = "https://www.huangdao.net/inc/yzm.php?r=0.8730639098210031?"


def request_captcha_image():
    return requests.get(captcha_image_url)


def label_to_code(predict_label):
    c0 = np.argmax(predict_label[0, 0:10].data.numpy())
    c1 = np.argmax(predict_label[0, 10:2 * 10].data.numpy())
    c2 = np.argmax(predict_label[0, 2 * 10:3 * 10].data.numpy())
    c3 = np.argmax(predict_label[0, 3 * 10:4 * 10].data.numpy())
    predict_label = '%s%s%s%s' % (
        ALL_CHARSET[c0],
        ALL_CHARSET[c1],
        ALL_CHARSET[c2],
        ALL_CHARSET[c3],
    )
    return predict_label
