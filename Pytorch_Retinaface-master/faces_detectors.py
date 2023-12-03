import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
import torch
from counters.detect import remove_prefix,check_keys
from models.retinaface import RetinaFace
from data import  cfg_re50
import cv2
import numpy as np
import time
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Processor')

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(1000, 400)
        # self.image_label.setAlignment(2)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.output_label = QLabel(self)
        self.output_label.setFixedSize(1000, 400)
        # self.output_label.setAlignment(2)
        self.output_label.setAlignment(Qt.AlignCenter)

        self.load_pic_button = QPushButton('加载图片', self)
        self.load_pic_button.setFixedSize(200,50)
        self.load_pic_button.move(610,210)
        # self.load_pic_button.setGeometry(210, 210, 200, 50)
        self.load_pic_button.clicked.connect(self.load_image)

        self.load_model_button =  QPushButton('加载模型', self)
        self.load_model_button.setGeometry(10,10,210,60)
        self.load_model_button.setFixedSize(200,50)
        self.load_model_button.clicked.connect(self.load_model)

        self.infer_button =  QPushButton('推理', self)
        self.infer_button.setFixedSize(200, 50)
        self.infer_button.setGeometry(10, 410, 200, 50)
        self.infer_button.clicked.connect(self.infer_and_show_faces)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.load_model_button)
        self.layout.addWidget(self.load_pic_button)

        self.layout.addWidget(self.infer_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.output_label)

        self.setLayout(self.layout)


    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap)
            self.process_image(filename)



    def load_model(self):
        self.cfg = cfg_re50
        self.model = RetinaFace(self.cfg, phase = 'test')
        self.confidence_threshold=0.02
        self.top_k = 5000
        self.keep_top_k = 750
        self.vis_thres = 0.6
        self.nms_threshold = 0.4

        pretrained_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Image Files (*.pth)')
        if pretrained_path:
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            check_keys(self.model, pretrained_dict)
            self.model.load_state_dict(pretrained_dict, strict=False)
            self.model.eval()
            print('Finished loading model!')
            self.model = self.model.to(self.device)
    def process_image(self, filename):

        self.img_raw = cv2.imread(filename, cv2.IMREAD_COLOR)

        self.img = np.float32(self.img_raw)

        self.im_height, self.im_width, _ = self.img.shape
        self.scale = torch.Tensor([self.img.shape[1], self.img.shape[0], self.img.shape[1], self.img.shape[0]])
        self.img -= (104, 117, 123)
        self.img = self.img.transpose(2, 0, 1)
        self.img = torch.from_numpy(self.img).unsqueeze(0)
        self.img = self.img.to(self.device)
        self.scale = self.scale.to(self.device)
        self.resize = 1


        # image = Image.open(filename)
        # 在这里可以添加处理图像的代码，例如对图像进行处理、分析等
        # 这里暂时将输出标签显示原始图像
        # output_pixmap = QPixmap.fromImage(QImage(filename))
        # self.output_label.setPixmap(output_pixmap)
    def infer_and_show_faces(self):
        tic = time.time()
        loc, conf, landms = self.model(self.img)
        print('net forward time: {:.4f}'.format(time.time() - tic))

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
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        for b in dets:
            if b[4] < self.vis_thres:
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
        # cv2.imshow('',self.img_raw)
        # cv2.waitKey()
        output_pixmap = QPixmap.fromImage(ImageProcessor.cvimg_to_qtimg(self.img_raw))
        # output_pixmap = QPixmap.fromImage(QImage(filename))
        self.output_label.setPixmap(output_pixmap)


    @staticmethod
    def qtpixmap_to_cvimg(qtpixmap):
        qimg = qtpixmap.toImage()
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]

        return result

    @staticmethod
    def cvimg_to_qtimg(cvimg):

        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

        return cvimg


def main():
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()