# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './GUI/Design.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1237, 919)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(9, 9, 1211, 661))
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
        self.label_19 = QtWidgets.QLabel(self.tab)
        self.label_19.setGeometry(QtCore.QRect(560, 510, 181, 41))
        self.label_19.setObjectName("label_19")
        self.interpolation_alg_centering = QtWidgets.QComboBox(self.tab)
        self.interpolation_alg_centering.setGeometry(QtCore.QRect(570, 560, 251, 23))
        self.interpolation_alg_centering.setObjectName("interpolation_alg_centering")
        self.interpolation_alg_centering.addItem("")
        self.interpolation_alg_centering.addItem("")
        self.interpolation_alg_centering.addItem("")
        self.interpolation_alg_centering.addItem("")
        self.interpolation_alg_centering.addItem("")
        self.interpolation_alg_centering.addItem("")
        self.interpolation_alg_centering.addItem("")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab_2)
        self.tabWidget_2.setGeometry(QtCore.QRect(20, 0, 1191, 411))
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.brute_rot = QtWidgets.QRadioButton(self.tab_4)
        self.brute_rot.setGeometry(QtCore.QRect(30, 120, 231, 32))
        self.brute_rot.setObjectName("brute_rot")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.brute_rot)
        self.fibonacci_rot = QtWidgets.QRadioButton(self.tab_4)
        self.fibonacci_rot.setGeometry(QtCore.QRect(30, 170, 271, 32))
        self.fibonacci_rot.setChecked(False)
        self.fibonacci_rot.setObjectName("fibonacci_rot")
        self.buttonGroup_2.addButton(self.fibonacci_rot)
        self.quadratic_rot = QtWidgets.QRadioButton(self.tab_4)
        self.quadratic_rot.setGeometry(QtCore.QRect(30, 210, 251, 61))
        self.quadratic_rot.setChecked(True)
        self.quadratic_rot.setObjectName("quadratic_rot")
        self.buttonGroup_2.addButton(self.quadratic_rot)
        self.groupBox = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox.setGeometry(QtCore.QRect(360, 20, 321, 201))
        self.groupBox.setObjectName("groupBox")
        self.ange_step_label = QtWidgets.QLabel(self.groupBox)
        self.ange_step_label.setGeometry(QtCore.QRect(20, 30, 121, 16))
        self.ange_step_label.setObjectName("ange_step_label")
        self.angle_step_1 = QtWidgets.QLineEdit(self.groupBox)
        self.angle_step_1.setGeometry(QtCore.QRect(140, 30, 71, 23))
        self.angle_step_1.setObjectName("angle_step_1")
        self.angle_step_2 = QtWidgets.QLineEdit(self.groupBox)
        self.angle_step_2.setGeometry(QtCore.QRect(140, 70, 71, 23))
        self.angle_step_2.setObjectName("angle_step_2")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(20, 70, 121, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(20, 100, 121, 31))
        self.label_9.setObjectName("label_9")
        self.angle_step_3 = QtWidgets.QLineEdit(self.groupBox)
        self.angle_step_3.setGeometry(QtCore.QRect(140, 110, 71, 23))
        self.angle_step_3.setObjectName("angle_step_3")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(20, 140, 141, 16))
        self.label_10.setObjectName("label_10")
        self.zoom1_ratio = QtWidgets.QLineEdit(self.groupBox)
        self.zoom1_ratio.setGeometry(QtCore.QRect(170, 140, 71, 23))
        self.zoom1_ratio.setObjectName("zoom1_ratio")
        self.zoom2_ratio = QtWidgets.QLineEdit(self.groupBox)
        self.zoom2_ratio.setGeometry(QtCore.QRect(170, 170, 71, 23))
        self.zoom2_ratio.setObjectName("zoom2_ratio")
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(20, 170, 141, 16))
        self.label_11.setObjectName("label_11")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_2.setGeometry(QtCore.QRect(700, 200, 321, 121))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setGeometry(QtCore.QRect(30, 30, 101, 16))
        self.label_12.setObjectName("label_12")
        self.precision_fib = QtWidgets.QLineEdit(self.groupBox_2)
        self.precision_fib.setGeometry(QtCore.QRect(140, 30, 71, 23))
        self.precision_fib.setObjectName("precision_fib")
        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        self.label_17.setGeometry(QtCore.QRect(30, 70, 131, 16))
        self.label_17.setObjectName("label_17")
        self.max_points_fib = QtWidgets.QLineEdit(self.groupBox_2)
        self.max_points_fib.setGeometry(QtCore.QRect(140, 70, 71, 23))
        self.max_points_fib.setObjectName("max_points_fib")
        self.label_20 = QtWidgets.QLabel(self.groupBox_2)
        self.label_20.setGeometry(QtCore.QRect(30, 100, 101, 16))
        self.label_20.setObjectName("label_20")
        self.cost_tolerance_fib = QtWidgets.QLineEdit(self.groupBox_2)
        self.cost_tolerance_fib.setGeometry(QtCore.QRect(170, 100, 71, 23))
        self.cost_tolerance_fib.setObjectName("cost_tolerance_fib")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_3.setGeometry(QtCore.QRect(700, 20, 321, 171))
        self.groupBox_3.setObjectName("groupBox_3")
        self.precision_quad = QtWidgets.QLineEdit(self.groupBox_3)
        self.precision_quad.setGeometry(QtCore.QRect(220, 50, 71, 23))
        self.precision_quad.setObjectName("precision_quad")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(30, 50, 101, 16))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.groupBox_3)
        self.label_16.setGeometry(QtCore.QRect(30, 90, 131, 16))
        self.label_16.setObjectName("label_16")
        self.max_it_quad = QtWidgets.QLineEdit(self.groupBox_3)
        self.max_it_quad.setGeometry(QtCore.QRect(220, 90, 71, 23))
        self.max_it_quad.setObjectName("max_it_quad")
        self.label_21 = QtWidgets.QLabel(self.groupBox_3)
        self.label_21.setGeometry(QtCore.QRect(30, 120, 161, 16))
        self.label_21.setObjectName("label_21")
        self.cost_tolerance_quad = QtWidgets.QLineEdit(self.groupBox_3)
        self.cost_tolerance_quad.setGeometry(QtCore.QRect(220, 120, 71, 23))
        self.cost_tolerance_quad.setObjectName("cost_tolerance_quad")
        self.label = QtWidgets.QLabel(self.tab_4)
        self.label.setGeometry(QtCore.QRect(30, 90, 251, 31))
        self.label.setObjectName("label")
        self.run_rotation_algorithm = QtWidgets.QPushButton(self.tab_4)
        self.run_rotation_algorithm.setGeometry(QtCore.QRect(430, 280, 121, 41))
        self.run_rotation_algorithm.setObjectName("run_rotation_algorithm")
        self.theta_max = QtWidgets.QLineEdit(self.tab_4)
        self.theta_max.setGeometry(QtCore.QRect(220, 20, 51, 23))
        self.theta_max.setObjectName("theta_max")
        self.theta_min = QtWidgets.QLineEdit(self.tab_4)
        self.theta_min.setGeometry(QtCore.QRect(150, 20, 41, 23))
        self.theta_min.setObjectName("theta_min")
        self.label_5 = QtWidgets.QLabel(self.tab_4)
        self.label_5.setGeometry(QtCore.QRect(30, 20, 121, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_4)
        self.label_6.setGeometry(QtCore.QRect(200, 20, 21, 16))
        self.label_6.setObjectName("label_6")
        self.label_13 = QtWidgets.QLabel(self.tab_4)
        self.label_13.setGeometry(QtCore.QRect(460, 320, 401, 101))
        self.label_13.setObjectName("label_13")
        self.label_18 = QtWidgets.QLabel(self.tab_4)
        self.label_18.setGeometry(QtCore.QRect(20, 270, 181, 41))
        self.label_18.setObjectName("label_18")
        self.interpolation_alg_angle = QtWidgets.QComboBox(self.tab_4)
        self.interpolation_alg_angle.setGeometry(QtCore.QRect(30, 320, 251, 23))
        self.interpolation_alg_angle.setObjectName("interpolation_alg_angle")
        self.interpolation_alg_angle.addItem("")
        self.interpolation_alg_angle.addItem("")
        self.interpolation_alg_angle.addItem("")
        self.interpolation_alg_angle.addItem("")
        self.interpolation_alg_angle.addItem("")
        self.interpolation_alg_angle.addItem("")
        self.interpolation_alg_angle.addItem("")
        self.initial_guess_delta = QtWidgets.QLineEdit(self.tab_4)
        self.initial_guess_delta.setGeometry(QtCore.QRect(240, 60, 71, 23))
        self.initial_guess_delta.setObjectName("initial_guess_delta")
        self.label_14 = QtWidgets.QLabel(self.tab_4)
        self.label_14.setGeometry(QtCore.QRect(30, 50, 211, 31))
        self.label_14.setObjectName("label_14")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.tabWidget_2.addTab(self.tab_5, "")
        self.use_converted_i203 = QtWidgets.QRadioButton(self.tab_2)
        self.use_converted_i203.setGeometry(QtCore.QRect(470, 610, 261, 21))
        self.use_converted_i203.setObjectName("use_converted_i203")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.use_converted_i203)
        self.use_converted_i607 = QtWidgets.QRadioButton(self.tab_2)
        self.use_converted_i607.setGeometry(QtCore.QRect(470, 590, 261, 21))
        self.use_converted_i607.setChecked(True)
        self.use_converted_i607.setObjectName("use_converted_i607")
        self.buttonGroup.addButton(self.use_converted_i607)
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(280, 590, 181, 31))
        self.label_7.setObjectName("label_7")
        self.use_current_images = QtWidgets.QRadioButton(self.tab_2)
        self.use_current_images.setGeometry(QtCore.QRect(320, 540, 291, 41))
        self.use_current_images.setChecked(False)
        self.use_current_images.setObjectName("use_current_images")
        self.buttonGroup.addButton(self.use_current_images)
        self.show_plots = QtWidgets.QRadioButton(self.tab_2)
        self.show_plots.setGeometry(QtCore.QRect(130, 550, 100, 21))
        self.show_plots.setAutoExclusive(False)
        self.show_plots.setObjectName("show_plots")
        self.output_plots = QtWidgets.QRadioButton(self.tab_2)
        self.output_plots.setGeometry(QtCore.QRect(20, 550, 100, 21))
        self.output_plots.setChecked(True)
        self.output_plots.setAutoExclusive(False)
        self.output_plots.setObjectName("output_plots")
        self.tabWidget.addTab(self.tab_2, "")
        self.log_text = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.log_text.setGeometry(QtCore.QRect(10, 690, 1191, 121))
        self.log_text.setObjectName("log_text")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1237, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Polarization by Conical Refraction - UAB Quantum and Atom Optics Group"))
        self.label_2.setText(_translate("MainWindow", "Image Directory:"))
        self.label_3.setText(_translate("MainWindow", "Output Data Directory:"))
        self.change_image_directory.setText(_translate("MainWindow", "Change Directory"))
        self.change_output_directory.setText(_translate("MainWindow", "Change Directory"))
        self.use_i607.setText(_translate("MainWindow", "Use i607 images"))
        self.use_i203.setText(_translate("MainWindow", "Use i203 images"))
        self.label_4.setText(_translate("MainWindow", "Choose Image Sizes for Computations:"))
        self.convert_selected_images.setText(_translate("MainWindow", "Convert Selected Images"))
        self.label_19.setText(_translate("MainWindow", "Affine Transformation\n"
"Interpolation Algorithm:"))
        self.interpolation_alg_centering.setItemText(0, _translate("MainWindow", "bicubic interpolation"))
        self.interpolation_alg_centering.setItemText(1, _translate("MainWindow", "bilinear interpolation"))
        self.interpolation_alg_centering.setItemText(2, _translate("MainWindow", "nearest neighbor interpolation"))
        self.interpolation_alg_centering.setItemText(3, _translate("MainWindow", "Lanczos interpolation over 8x8 neighborhood"))
        self.interpolation_alg_centering.setItemText(4, _translate("MainWindow", "Bit exact bilinear interpolation"))
        self.interpolation_alg_centering.setItemText(5, _translate("MainWindow", "Bit exact nearest neighbor interpolation"))
        self.interpolation_alg_centering.setItemText(6, _translate("MainWindow", "resampling using pixel area relation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Choose Output Directory and Images"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "K1 - Gradient Algorithm"))
        self.brute_rot.setText(_translate("MainWindow", "Brute Force Grid Search"))
        self.fibonacci_rot.setText(_translate("MainWindow", "Fibonacci 3 point search"))
        self.quadratic_rot.setText(_translate("MainWindow", "Quadratic Fit 3 point search"))
        self.groupBox.setTitle(_translate("MainWindow", "Parameters for Brute Force Search"))
        self.ange_step_label.setText(_translate("MainWindow", "Angle Step 1 (rad):"))
        self.angle_step_1.setText(_translate("MainWindow", "0.3"))
        self.angle_step_2.setText(_translate("MainWindow", "0.07"))
        self.label_8.setText(_translate("MainWindow", "Angle Step 2 (rad):"))
        self.label_9.setText(_translate("MainWindow", "Angle Step 3 (rad)\n"
"Precision:"))
        self.angle_step_3.setText(_translate("MainWindow", "0.005"))
        self.label_10.setText(_translate("MainWindow", "Zoom 1 Ratio (over 1):"))
        self.zoom1_ratio.setText(_translate("MainWindow", "0.4"))
        self.zoom2_ratio.setText(_translate("MainWindow", "0.2"))
        self.label_11.setText(_translate("MainWindow", "Zoom 2 Ratio (over 1):"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Parameters for Fibonacci Search"))
        self.label_12.setText(_translate("MainWindow", "Precision* (rad):"))
        self.precision_fib.setText(_translate("MainWindow", "0.01"))
        self.label_17.setText(_translate("MainWindow", "Maximum Points:"))
        self.max_points_fib.setText(_translate("MainWindow", "10000"))
        self.label_20.setText(_translate("MainWindow", "Cost tolerance:"))
        self.cost_tolerance_fib.setText(_translate("MainWindow", "0"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Parameters for Quadratic Search"))
        self.precision_quad.setText(_translate("MainWindow", "0.01"))
        self.label_15.setText(_translate("MainWindow", "Precision* (rad):"))
        self.label_16.setText(_translate("MainWindow", "Maximum Iterations:"))
        self.max_it_quad.setText(_translate("MainWindow", "10000"))
        self.label_21.setText(_translate("MainWindow", "Cost tolerance (relative):"))
        self.cost_tolerance_quad.setText(_translate("MainWindow", "0"))
        self.label.setText(_translate("MainWindow", "Select the Search Algorithm"))
        self.run_rotation_algorithm.setText(_translate("MainWindow", "RUN"))
        self.theta_max.setText(_translate("MainWindow", "np.pi"))
        self.theta_min.setText(_translate("MainWindow", "-np.pi"))
        self.label_5.setText(_translate("MainWindow", "Angle Range (rad):"))
        self.label_6.setText(_translate("MainWindow", "to"))
        self.label_13.setText(_translate("MainWindow", "*Remember that the precision will have a minimum depending\n"
" on the image quality and the minimum rotation arithmetics.\n"
"Noise or discretization can induce plateaus in the minimum"))
        self.label_18.setText(_translate("MainWindow", "Affine Transformation\n"
"Interpolation Algorithm:"))
        self.interpolation_alg_angle.setItemText(0, _translate("MainWindow", "bicubic interpolation"))
        self.interpolation_alg_angle.setItemText(1, _translate("MainWindow", "bilinear interpolation"))
        self.interpolation_alg_angle.setItemText(2, _translate("MainWindow", "nearest neighbor interpolation"))
        self.interpolation_alg_angle.setItemText(3, _translate("MainWindow", "Lanczos interpolation over 8x8 neighborhood"))
        self.interpolation_alg_angle.setItemText(4, _translate("MainWindow", "Bit exact bilinear interpolation"))
        self.interpolation_alg_angle.setItemText(5, _translate("MainWindow", "Bit exact nearest neighbor interpolation"))
        self.interpolation_alg_angle.setItemText(6, _translate("MainWindow", "resampling using pixel area relation"))
        self.initial_guess_delta.setText(_translate("MainWindow", "0.1"))
        self.label_14.setText(_translate("MainWindow", "Initial guess angle delta (rad)\n"
"used for Fibonacci and Quadratic:"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "K2 - Rotatation Algorithm"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), _translate("MainWindow", "MO - Radial Histogram Algorithm"))
        self.use_converted_i203.setText(_translate("MainWindow", " ./i203_converted_images"))
        self.use_converted_i607.setText(_translate("MainWindow", " ./i607_converted_images"))
        self.label_7.setText(_translate("MainWindow", "Use all Converted Images in \n"
"Output Data Directory:"))
        self.use_current_images.setText(_translate("MainWindow", "Use Selected Images (if still not\n"
"converted it will be done now)"))
        self.show_plots.setText(_translate("MainWindow", "Show Plots"))
        self.output_plots.setText(_translate("MainWindow", "Output Plots"))
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
