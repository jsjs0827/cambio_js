from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace,MobileNetV1
from utils.box_utils import decode, decode_landm
import time
import logging
from abc import ABCMeta,abstractmethod
from typing import Optional

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default=r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\pth_model\Retinaface_model_v2-20231202T083246Z-001\Retinaface_model_v2\Resnet50_Final.pth',   # default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class FacesCounter(metaclass=ABCMeta):

    def __init__(self,model_path:str,model_cfg:dict):
        # if model_cfg.name == 'Resnet50':
        #     self.cfg = cfg_re50
        #     self.model = RetinaFace(self.cfg, phase='test')
        #
        # elif model_cfg.name == 'mobilenet0.25':
        #     self.cfg = cfg_mnet
        #     self.model = MobileNetV1()
        # self.model = load_model()
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.keep_top_k = 750
        self.vis_thres = 0.6
        self.nms_threshold = 0.4
        self.model = None
        self.resize = 1
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        # if model_path:
        #     if torch.cuda.is_available():
        #         self.device = torch.cuda.current_device()
        #         pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(self.device))
        #     if "state_dict" in pretrained_dict.keys():
        #         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        #     else:
        #         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        #     check_keys(self.model, pretrained_dict)
        #     self.model.load_state_dict(pretrained_dict, strict=False)
        #     self.model.eval()
        #     self.model = self.model.to(self.device)
        #     print('Finished loading {} model!'.format(self.cfg.name))

    # @abstractmethod
    def __call__(self, path_or_root):
        if not self.check(path_or_root) and self.model:
            self.process_single_img(path_or_root)
            tic = time.time()
            loc, conf, landms = self.model(self.img)  # forward pass
            print('net forward time: {:.4f}'.format(time.time() - tic))
            save_path = path_or_root.replace('.' + self.pic_format, 'result.' + self.pic_format)
            self.show_results(loc, conf, landms, save_path)

    def check(self,img_or_folders:str):
        if os.path.exists(img_or_folders):
            if os.path.isfile(img_or_folders):
                if img_or_folders.endswith('jpeg'):
                    self.pic_format = 'jpeg'
                elif img_or_folders.endswith('png'):
                    self.pic_format = 'png'
                else:
                    self.pic_format = img_or_folders.split('.')[-1]
                return 0 # 0 for pic
            elif os.path.isdir(img_or_folders):
                pics = [os.path.join(img_or_folders,i) for i in os.listdir(img_or_folders)]
                assert all([pic.endswith('jpeg') or pic.endswith('png') for pic in pics]) ==  len(pics),"图片格式不统一"
                self.pic_format = pics[0].split('.')[-1]
                return 1 # 1 for folder

    def process_single_img(self,pic_path:str):
        self.img_raw = cv2.imread(pic_path, cv2.IMREAD_COLOR)
        img = np.float32(self.img_raw)
        self.im_height, self.im_width, _ = img.shape
        self.scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        self.img = torch.from_numpy(img).unsqueeze(0)
        if self.device!='cpu':
            self.img = self.img.to(self.device)
            self.scale = self.scale.to(self.device)

    def show_results(self,loc,conf,landms,save_path):
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
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
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

            cv2.imwrite(save_path, self.img_raw)

class RetinaFaceCounter(FacesCounter):

    def __init__(self,model_path:str,model_cfg:dict):
        super(RetinaFaceCounter,self).__init__(model_path,model_cfg)
        self.cfg = cfg_re50
        self.model = RetinaFace(self.cfg, phase='test')

        if model_path:
            if self.device!='cpu':
                pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(self.device))
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            check_keys(self.model, pretrained_dict)
            self.model.load_state_dict(pretrained_dict, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            print('Finished loading {} model!'.format(self.cfg['name']))



    # def __call__(self, path_or_root):
    #     if not self.check(path_or_root):
    #         self.process_single_img(path_or_root)
    #         tic = time.time()
    #         loc, conf, landms = self.model(self.img)  # forward pass
    #         print('net forward time: {:.4f}'.format(time.time() - tic))
    #         save_path = path_or_root.replace('.'+self.pic_format,'result.'+self.pic_format)
    #         self.show_results(loc, conf, landms,save_path)

class MobileNetV1FacesCounter(FacesCounter):

    def __init__(self,model_path:str,model_cfg:dict):
        super(MobileNetV1FacesCounter,self).__init__(model_path,model_cfg)
        self.cfg = cfg_mnet
        self.model = MobileNetV1()

        if model_path:
            if self.device!='cpu':
                pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(self.device))
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            check_keys(self.model, pretrained_dict)
            self.model.load_state_dict(pretrained_dict, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            print('Finished loading {} model!'.format(self.cfg['name']))

    # def __call__(self, path_or_root):
    #     if not self.check(path_or_root):
    #         self.process_single_img(path_or_root)
    #         tic = time.time()
    #         loc, conf, landms = self.model(self.img)  # forward pass
    #         print('net forward time: {:.4f}'.format(time.time() - tic))
    #         save_path = path_or_root.replace('.'+self.pic_format,'result.'+self.pic_format)
    #         self.show_results(loc, conf, landms,save_path)


if __name__ == '__main__':
    detector = RetinaFaceCounter(model_path=r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\pth_model\Retinaface_model_v2-20231202T083246Z-001\Retinaface_model_v2\Resnet50_Final.pth',
                                 model_cfg=cfg_re50)
    detector(r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\curve\test.jpg')
