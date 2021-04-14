# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './GUI/Design.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1237, 953)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(9, 9, 1221, 611))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(550, 30, 171, 71))
        self.label_2.setObjectName("label_2")
        self.image_directory = QtWidgets.QLineEdit(self.tab)
        self.image_directory.setGeometry(QtCore.QRect(550, 100, 651, 41))
        self.image_directory.setReadOnly(True)
        self.image_directory.setObjectName("image_directory")
        self.picture_dir_tree = QtWidgets.QTreeWidget(self.tab)
        self.picture_dir_tree.setGeometry(QtCore.QRect(15, 21, 501, 531))
        self.picture_dir_tree.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.picture_dir_tree.setIconSize(QtCore.QSize(35, 35))
        self.picture_dir_tree.setObjectName("picture_dir_tree")
        self.picture_dir_tree.headerItem().setText(0, "Choose one or multiple image files for calibration tests:")
        self.output_directory = QtWidgets.QLineEdit(self.tab)
        self.output_directory.setGeometry(QtCore.QRect(550, 260, 641, 41))
        self.output_directory.setText("")
        self.output_directory.setReadOnly(True)
        self.output_directory.setObjectName("output_directory")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(550, 190, 201, 71))
        self.label_3.setObjectName("label_3")
        self.change_image_directory = QtWidgets.QPushButton(self.tab)
        self.change_image_directory.setGeometry(QtCore.QRect(970, 170, 181, 34))
        self.change_image_directory.setObjectName("change_image_directory")
        self.change_output_directory = QtWidgets.QPushButton(self.tab)
        self.change_output_directory.setGeometry(QtCore.QRect(970, 330, 181, 34))
        self.change_output_directory.setObjectName("change_output_directory")
        self.use_i607 = QtWidgets.QRadioButton(self.tab)
        self.use_i607.setGeometry(QtCore.QRect(570, 410, 231, 32))
        self.use_i607.setChecked(True)
        self.use_i607.setObjectName("use_i607")
        self.use_i203 = QtWidgets.QRadioButton(self.tab)
        self.use_i203.setGeometry(QtCore.QRect(570, 460, 231, 32))
        self.use_i203.setObjectName("use_i203")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(550, 340, 341, 71))
        self.label_4.setObjectName("label_4")
        self.convert_selected_images = QtWidgets.QPushButton(self.tab)
        self.convert_selected_images.setGeometry(QtCore.QRect(850, 420, 251, 61))
        self.convert_selected_images.setObjectName("convert_selected_images")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab_2)
        self.tabWidget_2.setGeometry(QtCore.QRect(0, 0, 1221, 581))
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.tabWidget_2.addTab(self.tab_5, "")
        self.tabWidget.addTab(self.tab_2, "")
        self.log_text = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.log_text.setGeometry(QtCore.QRect(20, 630, 1191, 251))
        self.log_text.setObjectName("log_text")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1237, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Image Directory:"))
        self.label_3.setText(_translate("MainWindow", "Output Data Directory:"))
        self.change_image_directory.setText(_translate("MainWindow", "Change Directory"))
        self.change_output_directory.setText(_translate("MainWindow", "Change Directory"))
        self.use_i607.setText(_translate("MainWindow", "Use i607 images"))
        self.use_i203.setText(_translate("MainWindow", "Use i203 images"))
        self.label_4.setText(_translate("MainWindow", "Choose Image Sizes for Computations:"))
        self.convert_selected_images.setText(_translate("MainWindow", "Convert Selected Images"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Choose Output Directory and Images"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "K1 - Gradient Algorithm"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "K2 - Rotatation Algorithm"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), _translate("MainWindow", "MO - Radial Histogram Algorithm"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Measure Angle"))
        self.log_text.setPlainText(_translate("MainWindow", "> Choose images to work with\n"
"> Press Convert Selected Images to output them and leave them ready.\n"
"> Choose angle compuation algorithm and run it.\n"
"\n"
"Note that if \"Convert Selected Images\" is not clicked before running the algorithms, this will be done internally anyway.\n"
""))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

