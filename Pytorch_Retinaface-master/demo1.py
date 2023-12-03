from PyQt5 import QtCore, QtGui, QtWidgets

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
        self.pushButton.setText(_translate("MainWindow", "导入图片"))
        self.label_2.setText(_translate("MainWindow", "渲染后"))
        self.pushButton_2.setText(_translate("MainWindow", "开始渲染"))
# 主函数文件（命名为 Color.py)
import sys
# from ColorSplash import Ui_MainWindow
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
# import Core
picName = ''
picType = ''

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):

        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 两个按钮分别对应打开图片和渲染图片的功能
        self.pushButton.clicked.connect(self.openpic)
        self.pushButton_2.clicked.connect(self.Show)

        # 选择图片并展示
    def openpic(self):
        global picName, picType

        picName, picType = QFileDialog.getOpenFileName(self, " 打 开 图 片", "", " All Files(*)")#用户自主选择图片
        picType = picName.split("/")[-1]
        # 获取图片类型
        picType = picType.split(".")[1]
        if picType == "gif":
        # 使用 QMovie 函数加载动态图片
            gif = QMovie(picName)
        # 调整动态图片大小
            gif.setScaledSize(QSize(self.label_3.width(), self.label_3.height()))
        # 设置 cacheMode为CacheAll时表示GIF环
            gif.setCacheMode(QMovie.CacheAll)
            self.label_3.setMovie(gif)
        #开始播放
            gif.start()
        else:
        # 使用 QPixmap 加载图片并调整大小
            img = QPixmap(picName).scaled(self.label_3.width(), self.label_3.height())
            self.label_3.setPixmap(img)
        # 渲染图片并展示

    def Show(self):
        if picType == "gif":

        # 检测所选图片并渲染
        #     a = Core.detect_and_color_splash(gif_path=picName)
            gif = QMovie(a)
            gif.setScaledSize(QSize(self.label_4.width(), self.label_4.height()))
            gif.setCacheMode(QMovie.CacheAll)
            self.label_4.setMovie(gif)
            gif.start()
        else:
            # a = Core.detect_and_color_splash(image_path=picName)
            img = QPixmap(a).scaled(self.label_4.width(), self.label_4.height())
            self.label_4.setPixmap(img)
            self.label_5.setText("转换完成!！!")
if __name__ == '__main__':

    app = QApplication(sys.argv)
    mywin = MyWindow()
    mywin.show()
    sys.exit(app.exec_())