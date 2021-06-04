from GUI.Design_ui import *
from SOURCE.Image_Manager import *
from SOURCE.Polarization_Obtention_Algorithms import *
try:
    from SOURCE.Camera_Controler import *
    global disable_camera_functionality
    disable_camera_functionality=False
except:
    global diable_camera_functionality
    disable_camera_functionality=True
from glob import glob
import logging
import sys
import os
import cv2

# pyuic5 -x ./GUI/Design.ui -o ./GUI/Design_ui.py


class QPlainTextEditLogger_NonBlocking(logging.Handler, QtCore.QObject):
    """
        To log into the QPlainTextEdit widget directly in a
        non-blocking way: using signals and slots instead of directly changing
        the log text box
    """
    sigLog = QtCore.pyqtSignal(str)  # define the signal that will send the log messages

    def __init__(self, widget):
        logging.Handler.__init__(self)
        QtCore.QObject.__init__(self)
        self.widget=widget # text widget where to append the text when emitted
        self.widget.setReadOnly(True) # set the widget to readonly
        # connect the emission of the signal with the function to append text
        # the emission will emit a string
        self.sigLog.connect(self.widget.appendPlainText)

    def emit(self, logRecord):
        message = str(logRecord.getMessage())
        self.sigLog.emit(message)

class Worker(QtCore.QThread):
    def __init__(self, func, args):
        super(Worker, self).__init__()
        self.set_new_task_and_arguments(func, args)

    def set_new_task_and_arguments(self, func, args):
        self.func = func
        if isinstance(args, (list, tuple)):
            self.args = args
        else:
            self.args = (args,)

    def run(self):
        self.func(*self.args) # si pones *self.args se desacoplan todos los argumentos


class Polarization_by_Conical_Refraction(QtWidgets.QMainWindow, Ui_MainWindow):
    # Define Signals
    # Create the cv2 plotter signal. This is necessary to do because gui stuff (like the
    # cv2 calls to qt) cannot be handled from secondary threads. In a blocking non-threaded
    # version of the code this was not at all a problem.
    plotter_cv2 = QtCore.pyqtSignal(np.ndarray, int, str)
    # expecting array to plot, int with the time to waitKey and a string with the label to show

    # Create a progress bar updater signal
    barUpdate_Live = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        #self.showMaximized()

        # Adapt the font size to screen size
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        #print(" Screen size : "  + str(sizeObject.height()) + "x"  + str(sizeObject.width()))
        if(sizeObject.height()>1000):
            font = QtGui.QFont()
            font.setPointSize(13)
            self.setFont(font)
        else:
            self.resize(800, 400)
            self.showFullScreen()


        # Fullscreen shortcut
        self.FullScreenSc = QtWidgets.QShortcut(QtGui.QKeySequence('F11'), self)
        self.FullScreenSc.activated.connect(self.toggleFullScreen)
        # Quit shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(self.close)

        # connect plot signal to the plotting function. This way gui handles it when signal emision
        self.plotter_cv2.connect(self.show_cv2_image) #type=QtCore.Qt.BlockingQueuedConnection
        # and connect signal to progress bar update: now run "self.barUpdate_Live.emit(10)"
        self.barUpdate_Live.connect(self.progressBar_Life.setValue)

        # set the working directories
        self.master_directory = os.getcwd()
        self.image_directory.setText(self.master_directory+'/DATA/')
        self.output_directory.setText(self.master_directory+'/OUTPUT/')

        # set app icon
        self.setWindowIcon(QtGui.QIcon(self.master_directory+'/GUI/ICONS/croissant5.png'))

        # Initialize tree view of directories
        self.load_project_structure(self.master_directory+'/DATA/', self.picture_dir_tree)


        # Set up logging to use your widget as a handler
        log_handler = QPlainTextEditLogger_NonBlocking(self.log_text)
        # You can format what is printed to text box
        # connect with logger
        logging.getLogger().addHandler(log_handler)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        # avoid matplotlib logging to user GUI
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        # We connect the events with their actions

        # when image directory path is changed, the tree should display a new look
        self.image_directory.textChanged.connect(lambda:
            (  self.picture_dir_tree.clear(),
            self.load_project_structure(self.image_directory.text(), self.picture_dir_tree)))

        # When the user clicks to choose the image or output directory a prompt will show up
        self.change_image_directory.clicked.connect(lambda:
            self.choose_directory("Choose directory for Images",  self.image_directory))
        self.change_output_directory.clicked.connect(lambda:
            self.choose_directory("choose directory for Output", self.output_directory))

        # when the button to convert selected images is pressed, run it
        self.image_loader_initialized = False # checking whether an instance for the Image_Loader has been already initialized
        self.convert_selected_images.clicked.connect(
            self.initialize_Angle_Calculator_instance_convert_images)

        # when the button to execute an algorithm is clicked run it
        self.run_rotation_algorithm.clicked.connect(
            self.execute_rotation_algorithm)
        self.run_mirror_algorithm.clicked.connect(
            self.execute_mirror_algorithm)
        self.run_histogram_algorithm.clicked.connect(
            self.execute_histogram_algorithm)
        self.run_gradient_algorithm.clicked.connect(
            self.execute_gradient_algorithm)

        # When live test buttons are pressed execute stuff
        self.camera_initialized=False
        self.testCamera.clicked.connect(self.run_test_camera)
        self.grabReference.clicked.connect(self.run_grab_reference)
        self.stopCamera.clicked.connect(self.stop_camera)
        self.runCamera.clicked.connect(self.run_camera)

        # Initialize a worker for the hard tasks
        self.strong_worker = Worker( None, None)
        self.strong_worker.finished.connect(lambda:
            self.block_hard_user_interaction(True))

        # disbale camera functionalities if not available their drivers
        if disable_camera_functionality:
            self.testCamera.setEnabled(False)
            self.grabReference.setEnabled(False)

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_cv2_image(self, image_array, t, label):
        """
        Plots the image array and shows it for t milliseconds with window label "label".
        To be called when plotting signal is emitted!
        """
        cv2.imshow(label, image_array)
        ok = cv2.waitKey(t)
        cv2.destroyAllWindows()
        return ok


    def load_project_structure(self, startpath, tree):
        """
        Generates the tree view in the tree widget given the master path
        """
        for element in os.listdir(startpath):
            path_info = startpath + "/" + element
            parent_itm = QtWidgets.QTreeWidgetItem(tree, [os.path.basename(element)])
            if os.path.isdir(path_info):
                self.load_project_structure(path_info, parent_itm)
                parent_itm.setIcon(0, QtGui.QIcon(self.master_directory+'/GUI/ICONS/file_cabinet.ico'))
            elif path_info[-4:]==".tif":
                parent_itm.setIcon(0, QtGui.QIcon(self.master_directory+'/GUI/ICONS/photo.ico'))
            else:
                parent_itm.setIcon(0, QtGui.QIcon(self.master_directory+'/GUI/ICONS/file_text.ico'))

    def get_selected_file_paths(self, tree, path_till_tree):
        """
        Returns a list of paths selected by the user in the tree widget
                Notes for developer:
                .child(j).text(0)-> el j-esimo child su text
                .parent().text(0)-> el txt del (unico) parent
                si no hay child or parent t edevuelve None
        """
        files=[]
        for item in tree.selectedItems():
            path=''
            while(item!=None):
                path='/'+item.text(0)+path
                item=item.parent()
            files.append(path_till_tree[:-1]+path)
        return files

    def choose_directory(self, label, display_widget):
        """
            Prompts the user to choose a directory, this path will be saved in the text of
            the display_widget.
        """
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, label)
        display_widget.setText(folderpath)

    def choose_file(self, label, guess, extension, display_widget):
        """
            Prompts the user to choose a file, this path will be saved in the text of
            the display_widget.
        """
        filepath = QtWidgets.QFileDialog.getOpenFileName(self, label, guess, extension)
        display_widget.setText(filepath[0])

    def block_hard_user_interaction(self, state):
        """
            Blocks main execution buttons to user if state is false.
            If it is true, then these buttons are unlocked.
        """
        self.convert_selected_images.setEnabled(state)
        self.run_rotation_algorithm.setEnabled(state)
        self.run_gradient_algorithm.setEnabled(state)
        self.run_mirror_algorithm.setEnabled(state)
        self.run_histogram_algorithm.setEnabled(state)
        self.testCamera.setEnabled(state)
        self.grabReference.setEnabled(state)

    def initialize_Angle_Calculator_instance_convert_images(self):
        """
            Initializes an instance of the Image_Manager class and executes the
            conversion of the raw images to the selected mode i607 or i203. The output images
            will be saved in the output directory.

        """
        # Block everything to user
        self.block_hard_user_interaction(False)

        # save the mode chosen by the user
        self.mode = 607 if self.use_i607.isChecked() else 203
        # initialize instance
        self.image_loader = Image_Manager(self.mode,
            self.choose_interpolation_falg(self.interpolation_alg_centering))

        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._initialize_Angle_Calculator_instance_convert_images, []
        )
        self.strong_worker.start()

    def _initialize_Angle_Calculator_instance_convert_images(self):
        """
        Secondary thread stuff for initialize_Angle_Calculator_instance_convert_images function.
        """
        # import the images
        ret = self.image_loader.get_raw_images_to_compute(
            self.get_selected_file_paths(self.picture_dir_tree, self.image_directory.text()))
        if ret==1: # no valid images elected
            return 1
        # create directory for outputing the resulting converted images
        os.makedirs(self.output_directory.text()+f"/i{self.mode}_converted_images/", exist_ok=True)
        # convert the images
        self.image_loader.compute_raw_to_i607_or_i203(self.output_directory.text()+
                                                    f"/i{self.mode}_converted_images/")
        self.image_loader_initialized=True

    def execute_rotation_algorithm(self):
        """
        Executes the rotation algorithm according to the requirements of the user.
        """
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._execute_rotation_algorithm, []
        )
        self.strong_worker.start()

    def _execute_rotation_algorithm(self):
        """
            Thread stuff for the rotation algorithm execution scheduling.
        """
        ret = self._check_image_loader_ready()
        if ret==1: # failed!
            return 1
        logging.info(" Image loader ready!")
        # Initialize instance of Rotation Algorithm calculator
        rotation_algorithm = Rotation_Algorithm(self.image_loader,
            eval(self.theta_min_R.text()), eval(self.theta_max_R.text()),
            self.choose_interpolation_falg(self.interpolation_alg_opt),
            float(self.initial_guess_delta_rad.text()), self.use_exact_grav_R.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Rotation Algorithm...")
        if self.brute.isChecked():
            rotation_algorithm.brute_force_search(
                [float(self.angle_step_1_rad.text()), float(self.angle_step_2_rad.text()),
                 float(self.angle_step_3_rad.text())], [float(self.zoom1_ratio.text()),
                 float(self.zoom2_ratio.text())]
            )
        elif self.fibonacci.isChecked():
            rotation_algorithm.fibonacci_ratio_search(
                float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
                float(self.cost_tolerance_fib.text())
            )
        else:
            rotation_algorithm.quadratic_fit_search(
                float(self.precision_quad_rad.text()),
                int(self.max_it_quad.text()),
                float(self.cost_tolerance_quad.text())
            )
        logging.info(f"Found Polarization Angles in rad = {rotation_algorithm.angles}\nFound optimals in rad = {rotation_algorithm.optimals}\n\nPrecisions (rad) = {rotation_algorithm.precisions}\n\nTimes (s) = {rotation_algorithm.times}")

        if (self.output_plots.isChecked() and self.brute.isChecked()):
            rotation_algorithm.save_result_plots_brute_force(self.output_directory.text())
        else:
            rotation_algorithm.save_result_plots_fibonacci_or_quadratic(self.output_directory.text())

    def execute_mirror_algorithm(self):
        """
        Executes the mirror flip algorithm according to the requirements of the user.
        """
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._execute_mirror_algorithm, []
        )
        self.strong_worker.start()

    def _execute_mirror_algorithm(self):
        """
            Thread stuff for the mirror flip algorithm execution scheduling.
        """
        ret = self._check_image_loader_ready()
        if ret==1: # failed!
            return 1
        logging.info(" Image loader ready!")
        # Initialize instance of Rotation Algorithm calculator
        method="bin" if self.use_binning_M.isChecked() else "mask" if self.use_masking_M.isChecked() else "aff"
        mirror_algorithm = Mirror_Flip_Algorithm(self.image_loader,
            eval(self.theta_min_M.text()), eval(self.theta_max_M.text()),
            self.choose_interpolation_falg(self.interpolation_alg_opt),
            float(self.initial_guess_delta_rad.text()), method, self.left_vs_right_M.isChecked(), self.use_exact_grav_M.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Mirror Flip Algorithm...")
        if self.brute.isChecked():
            mirror_algorithm.brute_force_search(
                [float(self.angle_step_1_rad.text()), float(self.angle_step_2_rad.text()),
                 float(self.angle_step_3_rad.text())], [float(self.zoom1_ratio.text()),
                 float(self.zoom2_ratio.text())]
            )
        elif self.fibonacci.isChecked():
            mirror_algorithm.fibonacci_ratio_search(
                float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
                float(self.cost_tolerance_fib.text())
            )
        else:
            mirror_algorithm.quadratic_fit_search(
                float(self.precision_quad_rad.text()),
                int(self.max_it_quad.text()),
                float(self.cost_tolerance_quad.text())
            )
        logging.info(f"Found polarization angles in rad = {mirror_algorithm.angles}\nFound optimals in rad = {mirror_algorithm.optimals}\n\nPrecisions (rad) = {mirror_algorithm.precisions}\n\nTimes (s) = {mirror_algorithm.times}")

        if (self.output_plots.isChecked() and self.brute.isChecked()):
            mirror_algorithm.save_result_plots_brute_force(self.output_directory.text())
        else:
            mirror_algorithm.save_result_plots_fibonacci_or_quadratic(self.output_directory.text())

    def execute_gradient_algorithm(self):
        """
        Executes the gradient algorithm according to the requirements of the user.
        """
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._execute_gradient_algorithm, []
        )
        self.strong_worker.start()

    def _execute_gradient_algorithm(self):
        """
            Thread stuff for the gradient algorithm execution scheduling.
        """
        ret = self._check_image_loader_ready()
        if ret==1: # failed!
            return 1
        logging.info(" Image loader ready!")
        # Initialize instance of Rotation Algorithm calculator
        gradient_algorithm = Gradient_Algorithm(self.image_loader,
            eval(self.min_rad_G.text()), eval(self.max_rad_G.text()),
            float(self.initial_guess_delta_pix.text()),
            self.use_exact_grav_G.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Gradient Algorithm...")
        if self.brute.isChecked():
            gradient_algorithm.brute_force_search(
                [float(self.angle_step_1_pix.text()), float(self.angle_step_2_pix.text()),
                 float(self.angle_step_3_pix.text())], [float(self.zoom1_ratio.text()),
                 float(self.zoom2_ratio.text())]
            )
        elif self.fibonacci.isChecked():
            gradient_algorithm.fibonacci_ratio_search(
                float(self.precision_fib_pix.text()), int(self.max_points_fib.text()),
                float(self.cost_tolerance_fib.text())
            )
        else:
            gradient_algorithm.quadratic_fit_search(
                float(self.precision_quad_pix.text()),
                int(self.max_it_quad.text()),
                float(self.cost_tolerance_quad.text())
            )
        logging.info(f"Found optimal radii in pixels = {gradient_algorithm.optimals}\n\nPrecisions (rad) = {gradient_algorithm.precisions}\n\nTimes (s) = {gradient_algorithm.times}\n\nPolarization Angle (rad)={gradient_algorithm.angles}")

        if (self.output_plots.isChecked() and self.brute.isChecked()):
            gradient_algorithm.save_result_plots_brute_force(self.output_directory.text())
        else:
            gradient_algorithm.save_result_plots_fibonacci_or_quadratic(self.output_directory.text())



    def execute_histogram_algorithm(self):
        """
        Executes the histogram algorithm according to the requirements of the user.
        """
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._execute_histogram_algorithm, []
        )
        self.strong_worker.start()

    def _execute_histogram_algorithm(self):
        """
            Thread stuff for the histogram algorithm execution scheduling.
        """
        ret = self._check_image_loader_ready()
        if ret==1: # failed!
            return 1
        logging.info(" Image loader ready!")
        # Initialize instance of Rotation Algorithm calculator
        histogram_algorithm = Radial_Histogram_Algorithm(self.image_loader,
            self.use_exact_grav_H.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Histogram Algorithm...")
        if self.use_raw_idx_mask_H.isChecked():
            histogram_algorithm.compute_histogram_masking(
                float(self.angle_bin_size_H.text())
                )
            title="Masking_Histogram"
        elif self.use_raw_idx_bin_H.isChecked():
            histogram_algorithm.compute_histogram_binning(
                float(self.angle_bin_size_H.text())
                )
            title="Binning_Histogram"
        else:
            histogram_algorithm.compute_histogram_interpolate(
                float(self.angle_bin_size_H.text())
            )
            title="Interpolating_Histogram"
        if self.fit_cos_H.isChecked():
            histogram_algorithm.refine_by_cosine_fit()

        logging.info(f"Found optimal angles in rad = {histogram_algorithm.angles}\n\nPrecisions (rad) = {histogram_algorithm.precision}\n\nTimes (s) = {histogram_algorithm.times}")

        if (self.output_plots.isChecked()):
            histogram_algorithm.save_result_plots(self.output_directory.text(),
            title)



    def _check_image_loader_ready(self):
        """
        Checks whether images have already been converted and whether user wants them to
        be used or rather to import already converted images. To be used in secondary
        thread before the execution of any algorithm. Algorithms from secondary threads
        will call it.
        """
        if self.use_current_images.isChecked():
            # make sure there is an instance of image_loader initilized!
            if not self.image_loader_initialized:
                self.mode = 607 if self.use_i607.isChecked() else 203
                # initialize instance
                self.image_loader = Image_Manager(self.mode,
                    self.choose_interpolation_falg(self.interpolation_alg_centering))
                ret = self._initialize_Angle_Calculator_instance_convert_images()
                if ret==1:
                    return 1
        else: # use all the images in the output directory
            self.mode = 607 if self.use_converted_i607.isChecked() else 203
            self.image_loader = Image_Manager(self.mode,
                self.choose_interpolation_falg(self.interpolation_alg_centering))
            ret = self.image_loader.import_converted_images(
                sorted(glob(f"{self.output_directory.text()}/i{self.mode}_converted_images/*")))
            if ret==1:
                return 1



    def choose_interpolation_falg(self, combo_widget):
        flags = {"nearest neighbor interpolation":cv2.INTER_NEAREST,
            "bilinear interpolation":cv2.INTER_LINEAR,
            "bicubic interpolation":cv2.INTER_CUBIC,
            "resampling using pixel area relation":cv2.INTER_AREA,
            "Lanczos interpolation over 8x8 neighborhood":cv2.INTER_LANCZOS4,
            "Bit exact bilinear interpolation":cv2.INTER_LINEAR_EXACT,
            "Bit exact nearest neighbor interpolation":cv2.INTER_NEAREST_EXACT}
        return flags[combo_widget.currentText()]


    def run_test_camera(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._run_test_camera, []
        )
        self.strong_worker.start()

    def _run_test_camera(self):
        self.stopCamera.setEnabled(True)
        self.initialize_camera()
        self.camera.test_Camera()
        self.block_hard_user_interaction(True)
        self.stopCamera.setEnabled(False)

    def run_grab_reference(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._run_grab_reference, []
        )
        self.strong_worker.start()

    def _run_grab_reference(self):
        self.stopCamera.setEnabled(True)
        self.initialize_camera()
        self.camera.grab_and_fix_reference()
        self.block_hard_user_interaction(True)
        self.stopCamera.setEnabled(False)
        self.runCamera.setEnabled(True)

    def run_camera(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._run_camera, []
        )
        self.strong_worker.start()

    def _run_camera(self):
        self.stopCamera.setEnabled(True)
        self.initialize_camera()
        self.camera.take_and_process_frames()
        self.block_hard_user_interaction(True)
        self.stopCamera.setEnabled(False)

    def stop_camera(self):
        self.camera.stop_camera=True

    def initialize_camera(self):
        image_manager = Image_Manager(607 if self.use_i607.isChecked() else 203,
             self.choose_interpolation_falg(self.interpolation_alg_centering),
		mainThreadPlotter=self.plotter_cv2)
        if self.liveG.isChecked(): # gradient algorithm########################################
            angle_algorithm = Gradient_Algorithm(image_manager,
                eval(self.min_rad_G.text()), eval(self.max_rad_G.text()),
                float(self.initial_guess_delta_pix.text()),
                self.use_exact_grav_G.isChecked())
            if self.brute.isChecked():
                angle_function=lambda:angle_algorithm.brute_force_search(
                    [float(self.angle_step_1_pix.text()), float(self.angle_step_2_pix.text()),
                     float(self.angle_step_3_pix.text())], [float(self.zoom1_ratio.text()),
                     float(self.zoom2_ratio.text())]
                )
            elif self.fibonacci.isChecked():
                angle_function=lambda:angle_algorithm.fibonacci_ratio_search(
                    float(self.precision_fib_pix.text()), int(self.max_points_fib.text()),
                    float(self.cost_tolerance_fib.text())
                )
            else:
                angle_function=lambda:angle_algorithm.quadratic_fit_search(
                    float(self.precision_quad_pix.text()),
                    int(self.max_it_quad.text()),
                    float(self.cost_tolerance_quad.text())
                )

        elif self.liveR.isChecked(): # rotation algorithm##########################################
            angle_algorithm = Rotation_Algorithm(image_manager,
                eval(self.theta_min_R.text()), eval(self.theta_max_R.text()),
                self.choose_interpolation_falg(self.interpolation_alg_opt),
                float(self.initial_guess_delta_rad.text()), self.use_exact_grav_R.isChecked())

            if self.brute.isChecked():
                angle_function=lambda:angle_algorithm.brute_force_search(
                    [float(self.angle_step_1_rad.text()), float(self.angle_step_2_rad.text()),
                     float(self.angle_step_3_rad.text())], [float(self.zoom1_ratio.text()),
                     float(self.zoom2_ratio.text())]
                )
            elif self.fibonacci.isChecked():
                angle_function=lambda:angle_algorithm.fibonacci_ratio_search(
                    float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
                    float(self.cost_tolerance_fib.text())
                )
            else:
                angle_function=lambda:angle_algorithm.quadratic_fit_search(
                    float(self.precision_quad_rad.text()),
                    int(self.max_it_quad.text()),
                    float(self.cost_tolerance_quad.text())
                )

        elif self.liveH.isChecked(): # histogram algorithm####################################
            angle_algorithm = Radial_Histogram_Algorithm(image_manager,
                self.use_exact_grav_H.isChecked())
            if self.use_raw_idx_mask_H.isChecked():
                angle_function=lambda:angle_algorithm.compute_histogram_masking(
                    float(self.angle_bin_size_H.text())
                    )
            elif self.use_raw_idx_bin_H.isChecked():
                angle_function=lambda:angle_algorithm.compute_histogram_binning(
                    float(self.angle_bin_size_H.text())
                    )
            else:
                angle_function=lambda:angle_algorithm.compute_histogram_interpolate(
                    float(self.angle_bin_size_H.text())
                )
            #if self.fit_cos_H.isChecked():
                #histogram_algorithm.refine_by_cosine_fit() append this to the lambda function

        else: # mirror algoriothm #######################################################
            method="bin" if self.use_binning_M.isChecked() else "mask" if self.use_masking_M.isChecked() else "aff"
            angle_algorithm = Mirror_Flip_Algorithm(image_manager,
                eval(self.theta_min_M.text()), eval(self.theta_max_M.text()),
                self.choose_interpolation_falg(self.interpolation_alg_opt),
                float(self.initial_guess_delta_rad.text()), method, self.left_vs_right_M.isChecked(), self.use_exact_grav_M.isChecked())
            # Get arguments and run algorithm depending on the chosen stuff
            if self.brute.isChecked():
                angle_function=lambda:angle_algorithm.brute_force_search(
                    [float(self.angle_step_1_rad.text()), float(self.angle_step_2_rad.text()),
                     float(self.angle_step_3_rad.text())], [float(self.zoom1_ratio.text()),
                     float(self.zoom2_ratio.text())]
                )
            elif self.fibonacci.isChecked():
                angle_function=lambda:angle_algorithm.fibonacci_ratio_search(
                    float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
                    float(self.cost_tolerance_fib.text())
                )
            else:
                angle_function=lambda:angle_algorithm.quadratic_fit_search(
                    float(self.precision_quad_rad.text()),
                    int(self.max_it_quad.text()),
                    float(self.cost_tolerance_quad.text())
                )

        if self.use_PiCamera.isChecked():

            self.camera=Pi_Camera(angle_algorithm, angle_function,
                 float(self.referenceAngle.text()), int(self.imagesInChunks.text()),
                 image_manager,
                 self.saveLifeOutput.isChecked(),
                 self.output_directory.text(), self.barUpdate_Live,
                 int(self.pi_w.text()), int(self.pi_h.text()))

        else:
            self.camera=Basler_Camera()







if __name__ == "__main__":

    # Initialize and execute app
    app = QtWidgets.QApplication([])
    window = Polarization_by_Conical_Refraction()
    window.show()
    app.exec_()
