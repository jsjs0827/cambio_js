from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import traceback
from utils.box_utils import decode, decode_landm
import time
import logging
from abc import ABCMeta,abstractmethod
from collections import OrderedDict
import json
import logging
logging.basicConfig(filename='test.log',filemode='a',format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
import pathlib

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
class FacesCounter(metaclass=ABCMeta):

    def __init__(self,count_cfg ):
        self.count_cfg = {}
        self.count_cfg.update(vars(count_cfg))
        self.count_cfg = DictToObject(self.count_cfg)
        # self.confidence_threshold = 0.02
        # self.top_k = 5000
        # self.keep_top_k = 750
        # self.vis_thres = 0.6
        # self.nms_threshold = 0.4
        self.model = None
        self.resize = 1
        self.result = OrderedDict()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()


    # @abstractmethod
    def __call__(self, path_or_root):
        if not self.check(path_or_root) and self.model:
            self.process_single_img(path_or_root)
            tic = time.time()
            loc, conf, landms = self.model(self.img)  # forward pass
            print('net forward time: {:.4f}'.format(time.time() - tic))
            save_path = path_or_root.replace('.' + self.pic_format, 'result.' + self.pic_format)
            self.show_results(loc, conf, landms, save_path)
        else:
            for i,sing_pic in enumerate(self.pics):
                self.process_single_img(sing_pic)
                tic = time.time()
                loc, conf, landms = self.model(self.img)  # forward pass
                print('net forward time: {:.4f}'.format(time.time() - tic))
                save_path = str(sing_pic).replace('.' + self.pic_format, 'result.' + self.pic_format)
                self.show_results(loc, conf, landms, save_path)


    def faces(self,pic_path):
        if pic_path in self.result.keys():
            return self.result[pic_path]
        else:
            print('Do not infer {}'.format(pic_path))

    def check(self,img_or_folders:str):
        tmp_p = pathlib.Path(img_or_folders)
        if tmp_p.exists():
            if tmp_p.is_file():
                if img_or_folders.endswith('jpeg'):
                    self.pic_format = 'jpeg'
                elif img_or_folders.endswith('png'):
                    self.pic_format = 'png'
                else:
                    self.pic_format = img_or_folders.split('.')[-1]
                return 0  # 0 for pic
            elif tmp_p.is_dir():
                self.pics = [pathlib.PurePath.joinpath(i) for i in tmp_p.iterdir()]
                assert sum(
                    [pic.suffix =='.jpeg' or pic.suffix =='.png' or pic.suffix =='.jpg' for pic in self.pics]) == len(
                    self.pics), "图片格式不统一"
                self.pic_format = self.pics[0].suffix[1:]
                return 1  # 1 for folder


        # if os.path.exists(img_or_folders):
        #     if os.path.isfile(img_or_folders):
        #         if img_or_folders.endswith('jpeg'):
        #             self.pic_format = 'jpeg'
        #         elif img_or_folders.endswith('png'):
        #             self.pic_format = 'png'
        #         else:
        #             self.pic_format = img_or_folders.split('.')[-1]
        #         return 0 # 0 for pic
        #     elif os.path.isdir(img_or_folders):
        #         self.pics = [os.path.join(img_or_folders,i) for i in os.listdir(img_or_folders)]
        #         assert sum([pic.endswith('jpeg') or pic.endswith('png') or pic.endswith('jpg') for pic in self.pics]) ==  len(self.pics),"图片格式不统一"
        #         self.pic_format = self.pics[0].split('.')[-1]
        #         return 1 # 1 for folder

    def process_single_img(self,pic_path:str):
        try:
            self.pic_path = pic_path
            # self.img_raw = cv2.imread(pic_path, cv2.IMREAD_COLOR)
            self.img_raw = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
            img = np.float32(self.img_raw)
            self.im_height, self.im_width, _ = img.shape
            self.scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            self.img = torch.from_numpy(img).unsqueeze(0)
            if self.device!='cpu':
                self.img = self.img.to(self.device)
                self.scale = self.scale.to(self.device)
        except Exception as e:
            logging.error('process_single_img pic_path:{} failed'.format(pic_path))
            traceback.print_exc()

    def show_results(self,loc,conf,landms,save_path):
        try:
            priorbox = PriorBox(self.cfg, image_size=(self.im_height, self.im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * self.scale / self.resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([self.img.shape[3], self.img.shape[2], self.img.shape[3], self.img.shape[2],
                                   self.img.shape[3], self.img.shape[2], self.img.shape[3], self.img.shape[2],
                                   self.img.shape[3], self.img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / self.resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > self.count_cfg.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.count_cfg.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.count_cfg.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.count_cfg.keep_top_k, :]
            landms = landms[:self.count_cfg.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # show image
            self.faces_counter = 0
            if save_path:
                for b in dets:
                    if b[4] < self.count_cfg.vis_thres:
                        continue
                    self.faces_counter += 1
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(self.img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(self.img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(self.img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(self.img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(self.img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(self.img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(self.img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image
                # cv2.imshow('', img_raw)
                # cv2.waitKey()
                # name = "test.jpg"

                # cv2.imwrite(save_path, self.img_raw)
                cv2.imencode('.png', self.img_raw)[1].tofile(save_path)

            self.result[self.pic_path] = self.faces_counter
            print(os.path.basename(self.pic_path), self.result[self.pic_path])
        except:
            logging.error('Post processing {} failed'.format(self.pic_path))
            traceback.print_exc()


