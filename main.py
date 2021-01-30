#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/28 下午4:58
# @Author : zengwb


import logging.config
logging.config.fileConfig("./config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
import sys
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
import copy
from face_det import FaceDet
from face_align import FaceAlign
from face_masker import FaceMasker


def main():
    show_result = False
    image_path = './Data/test-data/test1.jpg'
    # image_path = './1.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    face_detector = FaceDet()
    face_align = FaceAlign()

    bboxs = face_detector(image)
    # print(bboxs)
    image_show = image.copy()
    face_lms = []
    for box in bboxs:
        # print(box)
        landmarks = face_align(image, box)
        # print(landmarks, landmarks.shape)
        lms = np.reshape(landmarks.astype(np.int32), (-1))
        # print(lms, lms.shape)
        face_lms.append(lms)
        cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        for (x, y) in landmarks.astype(np.int32):
            cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)
        if show_result:
            cv2.imshow('lms', image_show)
            cv2.waitKey(0)

    # face masker
    is_aug = True
    mask_template_name = '0.png'
    mask_template_name2 = '1.png'
    masked_face_path = 'test1_mask1.jpg'
    face_masker = FaceMasker(is_aug)
    # ======masked one face========
    new_image = face_masker.add_mask_one(image, face_lms[0], mask_template_name, mask_template_name)
    # imsave(mask_template_name, new_image)
    plt.imshow(new_image)
    plt.show()

    # masked two face
    # new_image = face_masker.add_mask_two(image, face_lms, mask_template_name, masked_face_path)
    # plt.imshow(new_image)
    # plt.show()








if __name__ == '__main__':
    # list = [1,2,3]
    # m = 9
    # for i in range(len(list)):
    #     list.append(m)
    #     m += 1
    #     print(list)
    # image_path = '/home/zengwb/Documents/FaceX-Zoo/Face_add_Mask/Data/test-data/test1.jpg'
    # img = cv2.imread(image_path)
    # i = np.clip(img, -1, 1)
    # cv2.imshow('s', i)
    # cv2.waitKey(0)
    # exit()
    main()