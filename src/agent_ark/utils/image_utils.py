
from pathlib import Path
import numpy as np
from PIL import Image
import base64
import io


def _deal_frame_array(arr: np.ndarray) -> np.ndarray:
    # arr 可为 (c,h,w) or (h,w,c) or (h,w)
    a = arr
    # 强制为 numpy
    a = np.asarray(a)
    # 处理 float -> uint8（假设范围 [0,1]）
    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).astype(np.uint8)
    elif a.dtype != np.uint8:
        try:
            a = a.astype(np.uint8)
        except Exception:
            pass

    # 将 channel-first (c,h,w) -> (h,w,c)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))
    return a


def _save_frame_array(arr: np.ndarray, path: Path) -> None:
    a = _deal_frame_array(arr)
    # a.ndim == 2 or a.ndim == 3 with channels already last
    if Image is not None:
        img = Image.fromarray(a)
        img.save(str(path), format="PNG")
    else:
        # 没有 PIL 时保存为 .npy 备份
        with open(str(path) + ".npy", "wb") as f:
            np.save(f, a)


def env_arr_to_pil_image(arr: np.ndarray) -> Image.Image:
    """将环境返回的图像数组转换为 PIL.Image 对象。"""
    a = _deal_frame_array(arr)
    return Image.fromarray(a)


#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pil_image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
