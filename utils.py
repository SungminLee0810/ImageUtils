import os
import sys
import cv2
import numpy as np

from glob import glob


def imgfmt_convert(img_path, tgt_path, tgt_fmt='jpg', jpg_qual=95):
    """Converts input image format to target image format.

        # Arguments
            img_path: String, a path of source image.
            tgt_path: String, a path of target image.
            tgt_fmt: String, one of the following:
                - 'jpg' (default), 'png', 'tiff', 'jpeg', 'JPEG', 'bmp', 'webp'.
            jpg_qual: int, 0~100, if the target image format is jpg or jpeg, jpg_qual controls the image quality.

        # Returns
            String, a full path of target image, including image name.
    """
    if tgt_fmt not in ['jpg', 'png', 'tiff', 'jpeg', 'JPEG', 'bmp', 'webp']:
        raise ValueError(f'{tgt_fmt} is not supported')

    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]
    if tmg_fmt in ['jpg', 'jpeg']:
        cv2.imwrite(f'{tgt_path}/{img_name}.{tgt_fmt}', img, params=[cv2.IMWRITE_JPEG_QUALITY, jpg_qual])
    else:
        cv2.imwrite(f'{tgt_path}/{img_name}.{tgt_fmt}', img)

    return f'{tgt_path}/{img_name}.{tgt_fmt}'
