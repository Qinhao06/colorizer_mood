import sys

from PyQt5.QtWidgets import QApplication
from qt_material import apply_stylesheet

from app import ImageViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='light_pink.xml')
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
