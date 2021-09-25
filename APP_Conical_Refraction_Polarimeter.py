#!/usr/bin/python3

from GUI.Design_ui import *
from SOURCE.Image_Manager import *
from SOURCE.Polarization_Obtention_Algorithms import *
from SOURCE.Theoretical_Ring_Simulator import *
global disable_gpu_functionality
global disable_PiCamera_functionality
global disable_BaslerCamera_functionality
import sys
try:
    from SOURCE.GPU_Classes import *
    disable_gpu_functionality=False
except:
    disable_gpu_functionality=True

try:
    from SOURCE.Camera_Controler_PiCamera import Pi_Camera
    disable_PiCamera_functionality=False
except:
    #print("Unexpected error:", sys.exc_info()[1])
    disable_PiCamera_functionality=True

try:
    from SOURCE.Camera_Controler_BaslerCamera import Basler_Camera
    disable_BaslerCamera_functionality=False
except:
    #print("Unexpected error:", sys.exc_info()[1])
    disable_BaslerCamera_functionality=True

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
            #self.showFullScreen()


        # Fullscreen shortcut
        self.FullScreenSc = QtWidgets.QShortcut(QtGui.QKeySequence('F11'), self)
        self.FullScreenSc.activated.connect(self.toggleFullScreen)
        # Quit shortcuts
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(self.close)


        # connect plot signal to the plotting function. This way gui handles it when signal emision
        self.plotter_cv2.connect(self.show_cv2_image, type=QtCore.Qt.BlockingQueuedConnection) #type=QtCore.Qt.BlockingQueuedConnection
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
        self.load_project_structure(self.master_directory+'/DATA/', self.reference_dir_tree)


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
            self.load_project_structure(self.image_directory.text(), self.picture_dir_tree),
            self.reference_dir_tree.clear(),
            self.load_project_structure(self.image_directory.text(), self.reference_dir_tree)
            ))


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
        self.run_SC.clicked.connect(
            self.execute_simulation_coordinate_descent_algorithm)
        self.run_SS.clicked.connect(
            self.execute_simulation_tracker_simplex_algorithm)

        # When live test buttons are pressed execute stuff
        self.Picamera_initialized=False
        self.testCamera.clicked.connect(self.run_test_camera)
        self.grabReference.clicked.connect(self.run_grab_reference)
        self.stopCamera.clicked.connect(self.stop_camera)
        self.runCamera.clicked.connect(self.run_camera)

        # When Simulation button is pressed do it
        self.createImages.clicked.connect(self.run_simulations)

        # When full test button is pressed do it
        self.run_full_test.clicked.connect(self.run_full_benchmark)


        # Initialize a worker for the hard tasks
        self.strong_worker = Worker( None, None)
        self.strong_worker.finished.connect(lambda:
            self.block_hard_user_interaction(True))

        # disbale camera functionalities if not available their drivers
        if disable_PiCamera_functionality and disable_BaslerCamera_functionality:
            self.testCamera.setEnabled(False)
            self.grabReference.setEnabled(False)

        elif disable_PiCamera_functionality:
            self.use_BaslerCamera.setChecked(True)
            self.use_PiCamera.setEnabled(False)

        elif disable_BaslerCamera_functionality:
            self.use_PiCamera.setChecked(FTrue)
            self.use_BaslerCamera.setEnabled(False)
        # else both are available

        # disable gpu functionalities if jax is not available
        if disable_gpu_functionality:
            self.use_GPU_S.setChecked(False)
            self.use_GPU_Sim_Th.setChecked(False)
            self.use_GPU_S.setEnabled(False)
            self.use_GPU_Sim_Th.setEnabled(False)

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
            elif path_info[-4:]==".tif" or path_info[-4:]==".png" or path_info[-4:]==".jpg":
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
            files.append(path_till_tree+path)
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
        self.createImages.setEnabled(state)
        self.run_full_test.setEnabled(state)
        self.run_SC.setEnabled(state)
        self.run_SS.setEnabled(state)

    def initialize_Angle_Calculator_instance_convert_images(self):
        """
            Initializes an instance of the Image_Manager class and executes the
            conversion of the raw images to the selected mode iX, i607 or i203. The output images
            will be saved in the output directory.

        """
        # Block everything to user
        self.block_hard_user_interaction(False)

        # save the mode chosen by the user
        self.mode = 607 if self.use_i607.isChecked() else 203 if self.use_i203.isChecked() else int(self.iX.text())

        # initialize instance for reference images and for problem images
        self.reference_loader = Image_Manager(self.mode,
            self.choose_interpolation_falg(self.interpolation_alg_centering))
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
        # import the reference images
        ret = self.reference_loader.get_raw_images_to_compute(
            self.get_selected_file_paths(self.reference_dir_tree, self.image_directory.text()))
        if ret==1: # no valid images selected
            return 1

        # import the problem images
        ret = self.image_loader.get_raw_images_to_compute(
            self.get_selected_file_paths(self.picture_dir_tree, self.image_directory.text()))
        if ret==1: # no valid images selected
            return 1
        # create directory for outputing the resulting converted images
        os.makedirs(self.output_directory.text()+f"/i{self.mode}_converted_images/References", exist_ok=True)
        # convert the images
        self.reference_loader.compute_raw_to_iX(self.output_directory.text()+
                                                    f"/i{self.mode}_converted_images/References/")
        self.image_loader.compute_raw_to_iX(self.output_directory.text()+
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
        rotation_algorithm = Rotation_Algorithm(self.reference_loader,
            eval(self.theta_min_R.text()), eval(self.theta_max_R.text()),
            self.choose_interpolation_falg(self.interpolation_alg_opt),
            float(self.initial_guess_delta_rad.text()), self.use_exact_grav_R.isChecked())
        self._run_angle_algorithm_and_reference_stuff(rotation_algorithm, self.__execute_rotation_algorithm, "Rotation")


    def __execute_rotation_algorithm(self, rotation_algorithm):
        """
        The proper execution of the rotation algorithm, this is abstracted
        since it must be called two times: one for the reference images and
        once for the problem images.
        """
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


    def _run_angle_algorithm_and_reference_stuff( self, angle_algorithm, __func_execute_algorithm, name):
        """
        This is a function that has been abstracted to avoid code repetition.
        It just handles the refernce and problem image angle computation and
        correct processing of the results
        """
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running "+name+" Algorithm on REFERENCES...")
        __func_execute_algorithm(angle_algorithm)
        angle_algorithm.set_reference_angle(float(self.referenceAngleTest.text()))
        angle_algorithm.process_obtained_angles(deg_or_rad=self.last_unit.currentIndex())
        # Show results (and save them if asked by user)
        if self.output_plots.isChecked():
            out=self.output_directory.text()+'/'+name+'_Algorithm/RESULTS/'
            os.makedirs( out+'/References/', exist_ok=True)
            self.reference_loader.plot_rings_and_angles(angle_algorithm.polarization, angle_algorithm.polarization_precision, output_path=out+'/References/', show=self.show_plots.isChecked(), unit=self.last_unit.currentText())

        logging.info(" Running "+name+" Algorithm on PROBLEM images...")
        angle_algorithm.reInitialize(self.image_loader)
        __func_execute_algorithm(angle_algorithm)
        angle_algorithm.process_obtained_angles(deg_or_rad=self.last_unit.currentIndex())
        # Show results (and save them if asked by user)
        if self.output_plots.isChecked():
            self.image_loader.plot_rings_and_angles(angle_algorithm.polarization, angle_algorithm.polarization_precision, output_path=out, show=self.show_plots.isChecked(), unit=self.last_unit.currentText())
        self.image_loader_initialized=False


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
        mirror_algorithm = Mirror_Flip_Algorithm(self.reference_loader,
            eval(self.theta_min_M.text()), eval(self.theta_max_M.text()),
            self.choose_interpolation_falg(self.interpolation_alg_opt),
            float(self.initial_guess_delta_rad.text()), method, self.left_vs_right_M.isChecked(), self.use_exact_grav_M.isChecked())
        self._run_angle_algorithm_and_reference_stuff(mirror_algorithm, self.__execute_mirror_algorithm, "Mirror")

    def __execute_mirror_algorithm(self, mirror_algorithm):
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
        gradient_algorithm = Gradient_Algorithm(self.reference_loader,
            eval(self.min_rad_G.text()), eval(self.max_rad_G.text()),
            float(self.initial_guess_delta_pix.text()),
            self.use_exact_grav_G.isChecked())
        self._run_angle_algorithm_and_reference_stuff(gradient_algorithm, self.__execute_gradient_algorithm, "Gradient")

    def __execute_gradient_algorithm(self, gradient_algorithm):
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
        histogram_algorithm = Radial_Histogram_Algorithm(self.reference_loader,
            self.use_exact_grav_H.isChecked())
        self._run_angle_algorithm_and_reference_stuff(histogram_algorithm, self.__execute_histogram_algorithm, "Histogram")

    def __execute_histogram_algorithm(self, histogram_algorithm):
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
                self.mode = 607 if self.use_i607.isChecked() else 203 if self.use_i203.isChecked() else int(self.iX.text())
                # initialize instance for reference images and for problem images
                self.reference_loader = Image_Manager(self.mode,
                    self.choose_interpolation_falg(self.interpolation_alg_centering))
                self.image_loader = Image_Manager(self.mode,
                    self.choose_interpolation_falg(self.interpolation_alg_centering))
                ret = self._initialize_Angle_Calculator_instance_convert_images()
                if ret==1:
                    return 1
        else: # use all the images in the output directory
            self.mode = 607 if self.use_i607.isChecked() else 203 if self.use_i203.isChecked() else int(self.iX.text())
            # initialize instance for reference images and for problem images
            self.reference_loader = Image_Manager(self.mode,
                self.choose_interpolation_falg(self.interpolation_alg_centering))
            self.image_loader = Image_Manager(self.mode,
                self.choose_interpolation_falg(self.interpolation_alg_centering))
            ret = self.image_loader.import_converted_images(
                sorted(glob(f"{self.output_directory.text()}/i{self.mode}_converted_images/*")))
            ret = self.reference_loader.import_converted_images(
                sorted(glob(f"{self.output_directory.text()}/i{self.mode}_converted_images/References/*")))
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

    def execute_simulation_coordinate_descent_algorithm(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._execute_simulation_coordinate_descent_algorithm, []
        )
        self.strong_worker.start()

    def _execute_simulation_coordinate_descent_algorithm(self):
        ret = self._check_image_loader_ready()
        if ret==1: # failed!
            return 1
        logging.info(" Image loader ready!")
        # Initialize instance of Simulation Coordinate descent Algorithm calculator
        simulator_cd_algorithm = Simulation_Coordinate_Descent_Algorithm(self.reference_loader,
            int(self.max_cycles.text()), None, None,
            eval(self.min_phi_S.text()), eval(self.max_phi_S.text()),
            eval(self.min_R0_S.text()), eval(self.max_R0_S.text()),
            eval(self.min_R0_mag_S.text()), eval(self.max_R0_mag_S.text()),
            float(self.initial_guess_delta_R0_SC.text()),
            float(self.initial_guess_delta_phi_SC.text()),
            float(self.initial_guess_delta_Z_SC.text()),
            n=float(self.n_S.text()), w0=float(self.w0_S.text()), a0=float(self.a0_S.text()),
            max_k=float(self.maxK_S.text()), num_k=int(self.numK_S.text()),
            nx=eval(self.nx_S.text()), xChunk=int(self.xChunks_S.text()),
            yChunk=int(self.yChunks_S.text()), gpu=self.use_GPU_S.isChecked(),
            min_radi_G=eval(self.min_rad_G_2.text()),
            max_radi_G=eval(self.max_rad_G_2.text()),
            use_exact_gravicenter_G=self.use_exact_grav_G_2.isChecked(),
            initial_guess_delta_G=float(self.initial_guess_G_2.text()))
        self._run_angle_algorithm_and_reference_stuff(simulator_cd_algorithm, self.__execute_simulation_coordinate_descent_algorithm, "Simulation_Coordinate_Descent")

    def __execute_simulation_coordinate_descent_algorithm(self, simulator_cd_algorithm):
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Simulation Coordinate Descent Algorithm...")
        if self.fibonacci_SC.isChecked():
            mname="Fibonacci_Search_"
            simulator_cd_algorithm.fibonacci_ratio_search(
                float(self.prec_R0_SC.text()),
                int(self.max_pt_R0_SC.text()),
                float(self.prec_phi_SC.text()),
                int(self.max_pt_phi_SC.text()),
                float(self.prec_Z_SC.text()),
                int(self.max_pt_Z_SC.text()),
                float(self.prec_R0_mag_SC.text()),
                int(self.max_pt_R0_mag_SC.text()),
                float(self.abs_cost_tol_SC.text()),
                float(self.precision_G_2.text()), int(self.max_it_G_2.text()),
                float(self.cost_tol_G_2.text())
            )

        else:
            mname="Quadratic_Search_"
            simulator_cd_algorithm.quadratic_fit_search(
                float(self.prec_R0_SC.text()),
                int(self.max_pt_R0_SC.text()),
                float(self.prec_phi_SC.text()),
                int(self.max_pt_phi_SC.text()),
                float(self.prec_Z_SC.text()),
                int(self.max_pt_Z_SC.text()),
                float(self.prec_R0_mag_SC.text()),
                int(self.max_pt_R0_mag_SC.text()),
                float(self.abs_cost_tol_SC.text()),
                float(self.precision_G_2.text()), int(self.max_it_G_2.text()),
                float(self.cost_tol_G_2.text())
            )
        logging.info(f"Times (s) = {simulator_cd_algorithm.times}\n\nFound optimal radii in pixels = {simulator_cd_algorithm.best_radii}\n\Z plane position (w0 units) = {simulator_cd_algorithm.best_zs}\n\nPolarization Angle (rad)={simulator_cd_algorithm.best_angles}\n\nNumber of Simulations required ={simulator_cd_algorithm.simulations_required}")

        if (self.output_plots.isChecked()):
            simulator_cd_algorithm.save_result_plots(self.output_directory.text(), mname)



    def execute_simulation_tracker_simplex_algorithm(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._execute_simulation_tracker_simplex_algorithm, []
        )
        self.strong_worker.start()

    def _execute_simulation_tracker_simplex_algorithm(self):
        pass



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
        self.tabWidget_3.setEnabled(True)


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
        self.tabWidget_3.setEnabled(False)

    def run_camera(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        self.runCamera.setEnabled(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._run_camera, []
        )
        self.strong_worker.start()

    def _run_camera(self):
        self.stopCamera.setEnabled(True)
        self.camera.take_and_process_frames(int(self.totalFrames.text()), int(self.outputEvery.text()))
        self.block_hard_user_interaction(True)
        self.stopCamera.setEnabled(False)
        self.runCamera.setEnabled(True)
        self.tabWidget_3.setEnabled(True)

    def stop_camera(self):
        self.camera.stop_camera=True

    def initialize_camera(self):
        image_manager = Image_Manager(607 if self.use_i607.isChecked() else 203 if self.use_i203.isChecked() else int(self.iX.text()),
             self.choose_interpolation_falg(self.interpolation_alg_centering),
		           mainThreadPlotter=self.plotter_cv2)
        self.mode = 607 if self.use_i607.isChecked() else 203 if self.use_i203.isChecked() else int(self.iX.text())

        if self.liveG.isChecked(): # gradient algorithm########################################
            angle_algorithm = Gradient_Algorithm(image_manager,
                eval(self.min_rad_G.text()), eval(self.max_rad_G.text()),
                float(self.initial_guess_delta_pix.text()),
                self.use_exact_grav_G.isChecked(), initialize_it=False)
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
                float(self.initial_guess_delta_rad.text()), self.use_exact_grav_R.isChecked(), initialize_it=False)

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
                self.use_exact_grav_H.isChecked(), initialize_it=False)
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
                float(self.initial_guess_delta_rad.text()), method, self.left_vs_right_M.isChecked(), self.use_exact_grav_M.isChecked(), initialize_it=False)
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
            if not self.Picamera_initialized:
                self.camera=Pi_Camera(angle_algorithm, angle_function,
                     float(self.referenceAngle.text()), int(self.imagesInChunks.text()),
                     image_manager,
                     self.saveLifeOutput.isChecked(),
                     self.output_directory.text(), self.barUpdate_Live,
                     self.last_unit_live.currentIndex(),
                     int(self.pi_w.text()), int(self.pi_h.text()))
                self.Picamera_initialized=True
            else:
                self.camera.reInitialize(angle_algorithm, angle_function,
                     float(self.referenceAngle.text()), int(self.imagesInChunks.text()),
                     image_manager,
                     self.saveLifeOutput.isChecked(),
                     self.output_directory.text(), self.barUpdate_Live,
                     self.last_unit_live.currentIndex(),
                     int(self.pi_w.text()), int(self.pi_h.text()))

        else:
            self.camera=Basler_Camera(angle_algorithm, angle_function,
                 float(self.referenceAngle.text()), int(self.imagesInChunks.text()),
                 image_manager,
                 self.saveLifeOutput.isChecked(),
                 self.output_directory.text(), self.barUpdate_Live,
                 self.last_unit_live.currentIndex(),
                 int(self.basler_w.text()), int(self.basler_h.text()),
                 int(self.offsetx.text()), int(self.offsety.text()),
                 int(self.max_buffer_num.text()),
                 self.plotter_cv2, float(self.frame_previs_ms.text()))

    def run_simulations(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._run_simulations, []
        )
        self.strong_worker.start()

    def _run_simulations(self):
        if self.use_GPU_Sim_Th.isChecked():
            simulator = RingSimulator_GPU( n=float(self.n.text()),
                w0=float(self.w0.text()), R0=float(self.R0.text()), a0=1.0,
                max_k=float(self.kmax.text()), num_k=int(self.nk.text()), nx=int(self.nx.text()),
                ny=int(self.ny.text()), nz=int(self.nz.text()), xmin=float(self.xmin.text()),
                xmax=float(self.xmax.text()),ymin=float(self.ymin.text()),ymax=float(self.ymax.text()),
                zmin=float(self.zmin.text()), zmax=float(self.zmax.text()),
                sim_chunk_x=int(self.sim_chunk_x.text()), sim_chunk_y=int(self.sim_chunk_y.text()) )

        else:
            simulator = RingSimulator( n=float(self.n.text()),
                w0=float(self.w0.text()), R0=float(self.R0.text()), a0=1.0,
                max_k=float(self.kmax.text()), num_k=int(self.nk.text()), nx=int(self.nx.text()),
                ny=int(self.ny.text()), nz=int(self.nz.text()), xmin=float(self.xmin.text()),
                xmax=float(self.xmax.text()),ymin=float(self.ymin.text()),ymax=float(self.ymax.text()),
                zmin=float(self.zmin.text()), zmax=float(self.zmax.text()),
                sim_chunk_x=int(self.sim_chunk_x.text()), sim_chunk_y=int(self.sim_chunk_y.text()) )

        inJones=np.array([eval(self.polar0.text()), eval(self.polar1.text())])


        if self.fullSimulation.isChecked():
            path=f"{self.output_directory.text()}/Simulated_Rings/Full"
            os.makedirs(path, exist_ok=True)
            simulator.compute_intensity_Trupin_and_Plot( inJones, path )
        else:
            path=f"{self.output_directory.text()}/Simulated_Rings/Approx"
            os.makedirs(path, exist_ok=True)
            simulator.compute_intensity_Todor_and_Plot(
                np.arctan2(inJones.real[1], inJones.real[0]), path )

    def run_full_benchmark(self):
        # Block everything to user
        self.block_hard_user_interaction(False)
        # Run worker for non-blocking computations
        self.strong_worker.set_new_task_and_arguments(
            self._run_full_benchmark, []
        )
        self.strong_worker.start()

    def _run_full_benchmark(self):
        ret = self._check_image_loader_ready()
        if ret==1: # failed!
            return 1
        logging.info(" Image loader ready!")
        # Prepare dictionary to register all the information
        benchu = {'Image Name':[], 'Algorithm':[], 'Time':[], 'Found Optimal':[], 'Optimal Precision':[], 'Found Polarization':[], 'Ground Truth Polarization':[], 'True Precision':[], 'Correct Decimals':[]}
        # image names
        image_names=self.image_loader.raw_images_names
        # Get Ground Truth Polarization angles if simulated else set to None
        ground_truths={}
        for name in image_names:
            try:
                ground_truths[name]=float(name.split('PolAngle_')[1].split('_')[0])
            except:
                ground_truths[name]=np.nan

        def num_of_zeros(n): # To count the number of zero decimals before non-zeros
            s = '{:.16f}'.format(n).split('.')[1]
            return len(s) - len(s.lstrip('0'))
        # Function to extract algorithm run information into benchmark dictionary
        def to_benchmark_dict(bench, alg, images, alg_name, ground_truths):
            for key, name in zip(alg.times.keys(), images):
                benchu['Image Name'].append(name)
                benchu['Algorithm'].append(alg_name)
                if type(alg.times[key]) is dict: # to treat brute force output stages
                    benchu['Time'].append(sum(list(alg.times[key].values())))
                    benchu['Found Optimal'].append(alg.optimals[key]['Stage_2'])
                    benchu['Optimal Precision'].append(alg.precisions[key]['Stage_2'])
                else:
                    benchu['Time'].append(alg.times[key])
                    benchu['Optimal Precision'].append(alg.precisions[key])
                    try:
                        benchu['Found Optimal'].append(alg.optimals[key])
                    except:
                        benchu['Found Optimal'].append(np.nan)

                benchu['Found Polarization'].append(alg.angles[key]/2)
                benchu['Ground Truth Polarization'].append(ground_truths[name])
                benchu['True Precision'].append(abs(abs(ground_truths[name])-abs(alg.angles[key]/2)))
                try:
                    benchu['Correct Decimals'].append(num_of_zeros(benchu['True Precision'][-1]))
                except:
                    benchu['Correct Decimals'].append(0)

        import pandas as pd
        import datetime
        # Prepare Directories
        stamp=datetime.datetime.now()
        path=self.output_directory.text()+f"/Full_Benchmark/{stamp}/"
        os.makedirs(path, exist_ok=True)

        # ALGORITHM G ###################################################################
        # Initialize instance of Gradient Algorithm calculator
        gradient_algorithm = Gradient_Algorithm(self.image_loader,
            eval(self.min_rad_G.text()), eval(self.max_rad_G.text()),
            float(self.initial_guess_delta_pix.text()),
            self.use_exact_grav_G.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Gradient Algorithm...")
        gradient_algorithm.brute_force_search(
            [float(self.angle_step_1_pix.text()), float(self.angle_step_2_pix.text()),
             float(self.angle_step_3_pix.text())], [float(self.zoom1_ratio.text()),
             float(self.zoom2_ratio.text())])
        gradient_algorithm.save_result_plots_brute_force(path)
        to_benchmark_dict(benchu, gradient_algorithm, image_names, "G - Brute Force", ground_truths)

        gradient_algorithm.reInitialize(self.image_loader)
        gradient_algorithm.fibonacci_ratio_search(
            float(self.precision_fib_pix.text()), int(self.max_points_fib.text()),
            float(self.cost_tolerance_fib.text())
        )
        gradient_algorithm.save_result_plots_fibonacci_or_quadratic(path)
        to_benchmark_dict(benchu, gradient_algorithm, image_names, "G - Fibonacci Ratio", ground_truths)

        gradient_algorithm.reInitialize(self.image_loader)
        gradient_algorithm.quadratic_fit_search(
            float(self.precision_quad_pix.text()),
            int(self.max_it_quad.text()),
            float(self.cost_tolerance_quad.text())
        )
        gradient_algorithm.save_result_plots_fibonacci_or_quadratic(path)
        to_benchmark_dict(benchu, gradient_algorithm, image_names, "G - Quadratic Fit", ground_truths)

        # ALGORITHM R ###################################################################
        # Initialize instance of Rotation Algorithm calculator
        rotation_algorithm = Rotation_Algorithm(self.image_loader,
            eval(self.theta_min_R.text()), eval(self.theta_max_R.text()),
            self.choose_interpolation_falg(self.interpolation_alg_opt),
            float(self.initial_guess_delta_rad.text()), self.use_exact_grav_R.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Rotation Algorithm...")
        rotation_algorithm.brute_force_search(
                [float(self.angle_step_1_rad.text()), float(self.angle_step_2_rad.text()),
                 float(self.angle_step_3_rad.text())], [float(self.zoom1_ratio.text()),
                 float(self.zoom2_ratio.text())]
            )
        rotation_algorithm.save_result_plots_brute_force(path)
        to_benchmark_dict(benchu, rotation_algorithm, image_names, "R - Brute Force", ground_truths)

        rotation_algorithm.reInitialize(self.image_loader)
        rotation_algorithm.fibonacci_ratio_search(
                float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
                float(self.cost_tolerance_fib.text())
            )
        rotation_algorithm.save_result_plots_fibonacci_or_quadratic(path)
        to_benchmark_dict(benchu, rotation_algorithm, image_names, "R - Fibonacci Ratio", ground_truths)

        rotation_algorithm.reInitialize(self.image_loader)
        rotation_algorithm.quadratic_fit_search(
                float(self.precision_quad_rad.text()),
                int(self.max_it_quad.text()),
                float(self.cost_tolerance_quad.text())
            )
        rotation_algorithm.save_result_plots_fibonacci_or_quadratic(path)
        to_benchmark_dict(benchu, rotation_algorithm, image_names, "R - Quadratic Fit", ground_truths)

        # ALGORITHM H #########################################################
        # Initialize instance of Histogram Algorithm calculator
        histogram_algorithm = Radial_Histogram_Algorithm(self.image_loader,
            self.use_exact_grav_H.isChecked())
        # Get arguments and run algorithm depending on the chosen stuff
        logging.info(" Running Histogram Algorithm...")
        """
        histogram_algorithm.compute_histogram_masking(
                float(self.angle_bin_size_H.text())
                )

        to_benchmark_dict(benchu, histogram_algorithm, image_names, "H - Masking Histogram", ground_truths)
        histogram_algorithm.save_result_plots(path, "Masking_Histogram")
        """
        histogram_algorithm.reInitialize(self.image_loader)
        histogram_algorithm.compute_histogram_binning(
                float(self.angle_bin_size_H.text())
            )
        to_benchmark_dict(benchu, histogram_algorithm, image_names, "H - Binning Histogram", ground_truths)
        histogram_algorithm.save_result_plots(path, "Binning_Histogram")

        #histogram_algorithm.reInitialize(self.image_loader)
        #histogram_algorithm.compute_histogram_interpolate(
            #float(self.angle_bin_size_H.text())
            #)
        #to_benchmark_dict(benchu, histogram_algorithm, image_names, "H - Interpolating Histogram", ground_truths)
        #histogram_algorithm.save_result_plots(path, "Interpolating_Histogram")

        #if self.fit_cos_H.isChecked():
        #    histogram_algorithm.refine_by_cosine_fit()

        # ALGORITHM M ###########################################################################
        # Initialize instance of Rotation Algorithm calculator

        for method, full_meth in zip(["mask", "bin", "aff"],["- Masking", "- Binning", " - Affine"]):
            mirror_algorithm = Mirror_Flip_Algorithm(self.image_loader,
                eval(self.theta_min_M.text()), eval(self.theta_max_M.text()),
                self.choose_interpolation_falg(self.interpolation_alg_opt),
                float(self.initial_guess_delta_rad.text()), method, self.left_vs_right_M.isChecked(), self.use_exact_grav_M.isChecked())
            # Get arguments and run algorithm depending on the chosen stuff
            logging.info(" Running Mirror Flip Algorithm...")
            mirror_algorithm.reInitialize(self.image_loader)
            mirror_algorithm.brute_force_search(
                    [float(self.angle_step_1_rad.text()), float(self.angle_step_2_rad.text()),
                     float(self.angle_step_3_rad.text())], [float(self.zoom1_ratio.text()),
                     float(self.zoom2_ratio.text())]
                )
            mirror_algorithm.save_result_plots_brute_force(path)
            to_benchmark_dict(benchu, mirror_algorithm, image_names, "M - Brute Force"+full_meth, ground_truths)

            mirror_algorithm.reInitialize(self.image_loader)
            mirror_algorithm.fibonacci_ratio_search(
                    float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
                    float(self.cost_tolerance_fib.text())
                )
            mirror_algorithm.save_result_plots_fibonacci_or_quadratic(path)
            to_benchmark_dict(benchu, mirror_algorithm, image_names, "M - Fibonacci Ratio"+full_meth, ground_truths)

            mirror_algorithm.reInitialize(self.image_loader)
            mirror_algorithm.quadratic_fit_search(
                    float(self.precision_quad_rad.text()),
                    int(self.max_it_quad.text()),
                    float(self.cost_tolerance_quad.text())
                )
            mirror_algorithm.save_result_plots_fibonacci_or_quadratic(path)
            to_benchmark_dict(benchu, mirror_algorithm, image_names, "M - Quadratic Fit"+full_meth, ground_truths)

        benchu=pd.DataFrame(benchu).sort_values(by=['Image Name', 'Algorithm'])
        logging.info(f"BENCHMARK RESULTS:\n{benchu}")
        benchu.to_csv(path+"/Benchmark.csv")
        benchu.to_excel(path+"/Benchmark.xlsx")
        self.plot_benchmark_results(benchu, image_names, path)


    def plot_benchmark_results(self, benchu, images, path):
        fig, ax = plt.subplots(1,1,figsize=(20,12))
        barWidth = 0.1
        for name in images:
            ax.set_title('Benchmark on Image:\n'+name+'\n')
            benchm = benchu[benchu["Image Name"]==name]
            r1 = np.arange(len(benchm["Time"]))

            ax.bar(r1, benchm["Time"], color='orange', width=barWidth, edgecolor='white', label='Time')
            ax.set_ylabel("Time (s)", color='orange')
            ax.tick_params(axis='y', colors='orange')

            ax2=ax.twinx()
            ax2.bar([x + barWidth for x in r1], benchm["Optimal Precision"], color='#557f2d', width=barWidth, edgecolor='white', label='Set Precision')
            ax2.bar([x + 2*barWidth for x in r1], benchm["True Precision"], color='blue', width=barWidth, edgecolor='white', label='True Precision')
            ax2.set_ylabel("Precision", color='blue')
            ax2.tick_params(axis='y', colors='blue')
            ax2.set_ylim((0,0.02))


            ax3=ax.twinx()
            ax3.bar([x + 3*barWidth for x in r1], benchm["Found Polarization"], color='#7f6d5f', width=barWidth, edgecolor='white', label='Found Polarization')
            ax3.bar([x + 4*barWidth for x in r1  ], benchm["Ground Truth Polarization"], color='brown', width=barWidth, edgecolor='white', label='G.True Polarization')
            ax3.tick_params(axis='y', colors='#7f6d5f')

            ax3.spines['right'].set_position(('axes', 1.07))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
            ax3.set_ylabel("Polarization (rad)", color='#7f6d5f')
            ax3.set_ylim((-np.pi/2, np.pi/2))

            ax4 = ax.twinx()
            ax4.bar([x + 5*barWidth for x in r1], benchm["Correct Decimals"], color='red', width=barWidth, edgecolor='white', label='Correct Decimals')
            ax4.yaxis.tick_left()
            rspine = ax4.spines['left']
            rspine.set_position(('axes', -0.07))
            ax4.yaxis.set_label_position("left")
            ax4.tick_params(axis='y', colors='red')
            ax4.set_ylim((0,10))
            #ax4.set_frame_on(True)
            #ax4.patch.set_visible(False)
            ax4.set_ylabel("Number of correct Decimals", color='red')

            fig.legend(loc="upper right")

            #ax.set_xlabel('Algorithm', fontweight='bold')
            plt.xticks([r + barWidth for r in range(len(benchm["Time"]))], benchm['Algorithm'])
            ax.tick_params(axis='x', rotation=30, length=10, width=2)
            ax3.grid(True,color='#7f6d5f')
            #ax2.grid(True, color='blue')
            #ax3.grid(True, color='orange')
            plt.savefig(path+f"/BENCH_{name}.png")
            ax.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            fig.clf()
            ax=fig.add_subplot(111)


if __name__ == "__main__":

    # Initialize and execute app
    app = QtWidgets.QApplication([])
    window = Polarization_by_Conical_Refraction()
    window.show()
    app.exec_()
