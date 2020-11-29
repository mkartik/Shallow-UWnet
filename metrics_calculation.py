# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from uiqm_utils import getUIQM

def calculate_metrics_ssim_psnr(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr = [], []

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)

        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)

        ground_truth_image = cv2.imread(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, resize_size)

        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, multichannel=True)
        error_list_ssim.append(error_ssim)

        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_list_psnr.append(error_psnr)

    return np.array(error_list_ssim), np.array(error_list_psnr)

def calculate_UIQM(image_path, resize_size=(256, 256)):
    image_list = os.listdir(image_path)
    uiqms = []

    for img in image_list:
        image = os.path.join(image_path, img)

        image = cv2.imread(image)
        image = cv2.resize(image, resize_size)

        # calculate UIQM
        uiqms.append(getUIQM(image))
    return np.array(uiqms)
