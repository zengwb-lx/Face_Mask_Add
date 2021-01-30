#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/28 下午5:00
# @Author : zengwb


import logging.config
logging.config.fileConfig("./config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import sys
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

with open('./config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)


class FaceDet():
    def __init__(self):
        model_dir = './models'
        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]
        logger.info('Start to load the face detection model...')
        try:
            faceDetModelLoader = FaceDetModelLoader(model_dir, model_category, model_name)
        except Exception as e:
            logger.info('Failed to parse model configuration file!')
            sys.exit(-1)
        else:
            logger.info('Successfully parsed the model configuration file model_meta.json!')

        try:
            self.model, self.cfg = faceDetModelLoader.load_model()
        except Exception as e:
            logger.error('Model loading failed!')
            sys.exit(-1)
        else:
            logger.info('Successfully loaded the face detection model!')

        self.faceDetModelHandler = FaceDetModelHandler(self.model, 'cuda:0', self.cfg)

    def __call__(self, image):
        # read image
        # image_path = '/home/zengwb/Documents/FaceX-Zoo/face_sdk/api_usage/test_images/test1.jpg'

        try:
            dets = self.faceDetModelHandler.inference_on_image(image)
        except Exception as e:
           logger.error('Face detection failed!')
           sys.exit(-1)
        else:
           logger.info('Successful face detection!')

        save_path_txt = './temp/test1_detect_res.txt'
        bboxs = dets
        # with open(save_path_txt, "w") as fd:
        #     for box in bboxs:
        #         line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
        #                str(int(box[2])) + " " + str(int(box[3])) + " " + \
        #                str(box[4]) + " \n"
        #         fd.write(line)
        #
        # for box in bboxs:
        #     box = list(map(int, box))
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # cv2.imwrite(save_path_img, image)
        logger.info('Successfully generate face detection results!')

        return bboxs


if __name__ == '__main__':
    image_path = '/home/zengwb/Documents/FaceX-Zoo/face_sdk/api_usage/test_images/test1.jpg'
    face_detector = FaceDet()
    bboxs = face_detector(image_path)
    print(bboxs)