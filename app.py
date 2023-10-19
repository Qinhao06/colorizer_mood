import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel, QLineEdit, QPushButton, QFileDialog, \
    QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QIcon
from qt_material import apply_stylesheet
import sys

from matplotlib import pyplot as plt

from colorizers import *


def get_img(file_name, select_option):

    # load colorizers
    if select_option == 'cheerful':
        colorizer_siggraph17 = torch.load('model_cheerful.pth').cpu().eval()
    elif select_option == 'horror':
        colorizer_siggraph17 = torch.load('model_horror.pth').cpu().eval()
    elif select_option == 'melancholy':
        colorizer_siggraph17 = torch.load('model_melancholy.pth').cpu().eval()
    elif select_option == 'romantic':
        colorizer_siggraph17 = torch.load('model_romantic.pth').cpu().eval()
    else:
        colorizer_siggraph17 = siggraph17(pretrained=True)

    # 图像处理
    img = load_img(file_name)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())




    # 转换 ndarray 为 QImage
    # plt.imsave('eccv16.png', out_img_eccv16)
    plt.imsave('siggraph17.jpg', out_img_siggraph17)
    plt.imsave('img_bw.jpg', img_bw)
    out_file_name = './siggraph17.jpg'
    pixmap = QPixmap(out_file_name)
    out_file_name = 'img_bw.jpg'
    pixmap2 = QPixmap(out_file_name)
    return pixmap, pixmap2


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon('icon.png'))

        # Create widgets
        self.combo_box = QComboBox()
        self.combo_box.addItems(["cheerful", "horror", "melancholy", "romantic"])

        self.label1 = QLabel()
        self.label2 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.label2.setAlignment(Qt.AlignCenter)

        # self.edit = QLineEdit()
        self.button = QPushButton("Upload")
        self.button.clicked.connect(self.upload_image)

        # Create layout
        layout = QVBoxLayout()

        # Add widgets to layout
        layout.addWidget(self.combo_box)
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)
        # layout.addWidget(self.edit)
        layout.addWidget(self.button)



        # Set layout for the main window
        self.setLayout(layout)
        self.resize(800, 600)

        # Set window title
        self.setWindowTitle("Image Viewer")

    def upload_image(self):
        # get combox

        # Open file dialog to select image file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp *.jpeg)", options=options)

        if file_name:
            select_option = self.combo_box.currentText()
            width = 600
            height = 300
            # Load image and display it in the label
            pixmap = QPixmap(file_name)
            pixmap2, img_bw = get_img(file_name, select_option)
            self.label1.setPixmap(img_bw.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.label2.setPixmap(
                pixmap2.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # 设置图片大小和长宽比


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='light_pink.xml')
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
