import os
import sys
import cv2
import json
import numpy as np

from glob import glob


def imgfmt_convert(img_path, tgt_path, tgt_fmt='jpg', jpg_qual=95):
    """Converts input image format to target image format.

        # Arguments
            img_path: String, a path to source image.
            tgt_path: String, a path to target image.
            tgt_fmt: String, one of the following:
                - 'jpg' (default), 'png', 'tiff', 'jpeg', 'JPEG', 'bmp', 'webp'.
            jpg_qual: int, 0~100, if the target image format is jpg or jpeg, jpg_qual controls the image quality.

        # Returns
            String, a full path to target image, including image name.
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


def PrecisionRecall_Calc(confusion_matrix):
    """Computes the precision, recall and F1-score from the confusion matrix.

        # Arguments
            confusion_matrix: 2D array, `[TP, FP, TN, FN]`.

        # Returns
            results: Dictionary, calculation results.
                - precision: (TP) / (TP + FP)
                - recall: (TP) / float(TP + FN)
                - F1-score: 2 * (precision * recall) / (precision + recall)
    """
    results = {}
    TP, FP, TN, FN = confusion_matrix
    results['precision'] = (TP) / float(TP + FP)
    results['recall'] = (TP) / float(TP + FN)
    results['F1-score'] = 2 * (results['precision'] * results['recall']) / float(
        results['precision'] + results['recall'])
    return results


def ExtractPatchFromCOCOJSON(json_path, imgdir_path, patch_shape, margin=0, save_imgpath=None, output_path=None):
    """Extracts the target object patches from a COCO style JSON file containing the bbox ground truth.

            # Arguments
                json_path: String, a path to JSON file.
                    the JSON file must be in COCO style format and contain the bbox GT.
                imgdir_path: String a path to source image directory.
                patch_shape: 2D array with shape `[W, H]`.
                    The shape of the extracted patch.
                    - W: Patch width.
                    - H: Patch height.
                margin: A scalar, An argument to give extracted patch a pixel margin.
                    Margins are applied in four directions: top, bottom, left, and right. (default: 0)
                save_imgpath: String, a path to save cropped patch image. (default: None)
                    If save_img is 'None', no image is saved.
                output_path: String, a path to save the output as .npz file. (default: None)
                    If save_path is 'None', no output file is saved.

            # Returns
                output: Dictionary, patch images and labels.
                - patch: 4D array with shape `[N, W, H, C]`
                    - N: The number of patches.
                    - W: Patch width.
                    - H: Patch height.
                    - C: A channel of patch.
                - label: 2D array with shape `[N x nClass]`
    """
    output = {}
    pheight, pwidth = patch_shape
    with open(json_path) as f:
        json_object = json.load(f)
    num_class = len(json_object['categories'])
    onehot_template = np.eye(num_class, num_class).astype(np.float32)
    # num_bboxes = len(json_object['annotations'])
    # cropped_patches = np.zeros((num_bboxes, height, width,), dtype=np.uint8)
    cropped_patches = []
    patch_labels = []
    for images_object in json_object['images']:
        img_name = images_object['file_name']
        img = cv2.imread(f'{imgdir_path}/{img_name}')
        img_height, img_width, img_ch = img.shape
        for anno_object in json_object['annotations']:
            if anno_object['image_id'] == images_object['id']:
                anno_id = anno_object['id']
                gt_x, gt_y, gt_w, gt_h = anno_object['bbox']
                x1 = max(gt_x - margin, 0)
                y1 = max(gt_y - margin, 0)
                x2 = min(gt_x + gt_w + margin, img_width)
                y2 = min(gt_y + gt_h + margin, img_height)
                cropped_patch = img[y1:y2, x1:x2, :]
                if save_imgpath is not None:
                    cv2.imwrite(f'{save_imgpath}/{anno_id}.png', cropped_patch)
                cropped_patches.append(cv2.resize(cropped_patch, (pheight, pwidth), interpolation=cv2.INTER_LINEAR))
                patch_labels.append(onehot_template[anno_object['category_id'] - 1])
    output['patch'] = np.array(cropped_patches)
    output['label'] = np.array(patch_labels)
    if output['patch'].ndim < 4:  # to match output shape `[N, W, H, C]`
        output['patch'] = np.expand_dims(output['patch'], axis=-1)
    if output_path is not None:
        np.savez(output_path, data=output['patch'], label=output['label'])
    return output
