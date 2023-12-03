from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import torch
from counters.detect import remove_prefix,check_keys
from models.retinaface import RetinaFace
from data import cfg_re50
import cv2
import numpy as np
import time
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


class Ui_MainWindow(object) :
    #设置 Ui_MainWindow 类
    def setupUi(self, MainWindow):
        MainWindow. setObjectName("MainWindow")
        #窗口名称
        MainWindow. resize(800, 600)
        #窗口大小
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self. centralwidget. setObjectName("centralwidget")
        #标签1和标签2的参数设置,用于说明图片
        self.label = QtWidgets. QLabel(self.centralwidget)
        self.label. setGeometry(QtCore. QRect(150, 450, 101, 41))
        font = QtGui. QFont( )
        font. setFamily("微软雅黑")
        font. setPointSize(13)
        self. label. setFont(font)
        self. label. setObjectName("label")
        self.label_2 = QtWidgets. QLabel(self.centralwidget)

        self.label_2.setGeometry(QtCore.QRect(530, 450, 71, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        # 按钮 1 和按钮 2参数设置用于连接函数实现功能
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 60, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 450, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(340, 90, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")

        # 标签 3 和标签 4 的参数设置，用于展示图片
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 180, 350, 250))
        font = QtGui.QFont()
        self.label_3.setFont(font)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(420, 180, 350, 250))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        # 标签5的参数设置,用于说明进程
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(320, 490, 161, 41))
        font = QtGui.QFont()
        font.setFamily("04b_21")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate

        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # 标签和按钮的内容
        self.label.setText(_translate("MainWindow", "原始图片"))
        self.pushButton.setText(_translate("MainWindow", "导入模型"))
        self.label_2.setText(_translate("MainWindow", "推理结果"))
        self.pushButton_2.setText(_translate("MainWindow", "开始推理"))
        self.pushButton_3.setText(_translate("MainWindow", "加载图片"))




class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):

        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 两个按钮分别对应打开图片和渲染图片的功能
        self.pushButton_3.clicked.connect(self.load_image)
        self.pushButton.clicked.connect(self.load_model)
        self.pushButton_2.clicked.connect(self.infer_and_show_faces)

        # self.pushButton_2.clicked.connect(self.Show)


        # 选择图片并展示
    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            # cv_img =   cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
            # cv_img = cv2.resize(cv_img,(161,41))
            # pixmap = QPixmap.fromImage(MyWindow.cvimg_to_qtimg(cv_img))
            pixmap = QPixmap(filename)

            self.label_3.setPixmap(pixmap)
            self.process_image(filename)

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
        output_pixmap = QPixmap.fromImage(MyWindow.cvimg_to_qtimg(self.img_raw))
        # output_pixmap = QPixmap.fromImage(QImage(filename))
        self.label_4.setPixmap(output_pixmap)

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


if __name__ == '__main__':

    app = QApplication(sys.argv)
    mywin = MyWindow()
    mywin.show()
    sys.exit(app.exec_())