import sys
import PyQt5
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

class MainUi(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        super(MainUi,self).__init__()
        self.init_ui()

    def init_ui(self):
        #可视化窗口大小
        self.setFixedSize(960,700)
        self.main_widget  = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(self.left_layout)
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setObjectName('right_widget')
        self.right_layout =  QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)

        self.main_layout.addWidget(self.left_widget,0,0,13,3)

        self.main_layout.addWidget(self.right_widget,0,5,13,7)

        self.setCentralWidget(self.main_widget)
        #左侧布局
        self.left_label_1 = QtWidgets.QLabel("人脸计数")
        self.left_label_1.setObjectName('left_label')

        #创建按钮
        self.left_button_1  = QtWidgets.QPushButton()
        self.left_button_1.setObjectName('left_button')
        self.left_button_1.clicked.connect(self.openimage)  # 事件绑定

        self.setWindowOpacity(1)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        self.main_widget.setStyleSheet('''
        QWidget# left_widget{
        background:gray;
        border-top:1px solid white;
        border-bottom:1px solid white;
        border-left:1px solid white;
        border-top-left-radius:10px;
        border-bottom-left-radius:10px;
        }
        ''')
        self.main_layout.setSpacing(2)

    def openimage(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap)
            self.process_image(filename)

def ShowUI():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.setObjectName('MainWindow')
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    ShowUI()