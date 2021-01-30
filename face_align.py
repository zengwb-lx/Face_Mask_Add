#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/28 下午5:27
# @Author : zengwb

import sys
import logging.config
logging.config.fileConfig("./config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('./config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)


class FaceAlign():
    def __init__(self):
        model_dir = './models'
        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_alignment'
        model_name = model_conf[scene][model_category]
        logger.info('Start to load the face landmark model...')
        # load model
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_dir, model_category, model_name)
        except Exception as e:
            logger.info('Failed to parse model configuration file!')
            sys.exit(-1)
        else:
            logger.info('Successfully parsed the model configuration file model_meta.json!')

        try:
            model, cfg = faceAlignModelLoader.load_model()
        except Exception as e:
            logger.error('Model loading failed!')
            sys.exit(-1)
        else:
            logger.info('Successfully loaded the face landmark model!')

        self.faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    def __call__(self, image, bbox):
        det = np.asarray(list(map(int, bbox[0:4])), dtype=np.int32)
        # print(det)
        landmarks = self.faceAlignModelHandler.inference_on_image(image, det)
        # print(landmarks)
        # image_show = image.copy()
        # for (x, y) in landmarks.astype(np.int32):
        #     cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)
        # cv2.imshow('lms', image_show)
        # cv2.waitKey(0)
        return landmarks


if __name__ == '__main__':
    face_align = FaceAlign()
    image_path = '/home/zengwb/Documents/FaceX-Zoo/face_sdk/api_usage/temp/test1_detect_res.jpg'
    image_det_txt_path = '/home/zengwb/Documents/FaceX-Zoo/face_sdk/api_usage/temp/test1_detect_res.txt'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    with open(image_det_txt_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        landmarks = face_align(image, line)


