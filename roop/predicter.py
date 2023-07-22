import numpy
import opennsfw2  # 导入 opensfw2 库
from PIL import Image

from roop.typing import Frame

MAX_PROBABILITY = 1


def predict_frame(target_frame: Frame) -> bool:
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    model = opennsfw2.make_open_nsfw_model()
    views = numpy.expand_dims(image, axis=0)
    # 下面这行代码注释掉，表示不进行检测
    # _, probability = model.predict(views)[0]
    return False  # 直接返回假值，表示不检测或不起作用


def predict_image(target_path: str) -> bool:
    # 下面这行代码注释掉，表示不进行检测
    # return opennsfw2.predict_image(target_path) > MAX_PROBABILITY
    return False  # 直接返回假值，表示不检测或不起作用


def predict_video(target_path: str) -> bool:
    # 下面这行代码注释掉，表示不进行检测
    # _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return False  # 直接返回假值，表示不检测或不起作用