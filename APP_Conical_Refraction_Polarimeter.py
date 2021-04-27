from GUI.Design_ui import *
from SOURCE.Polarization_Angle_Calculator import *
from glob import glob
import logging
import sys
import os

# pyuic5 -x ./GUI/Design.ui -o ./GUI/Design_ui.py

"""
    TODOs
    -----
    > Make a function that blocks all the stuff and unblocks it (all the buttons etc)
    > Make a non-blocking client perhaps

"""


class QPlainTextEditLogger_NonBlocking(logging.Handler, QtCore.QObject):
    """
        To log into the QPlainTextEdit widget directly in a
        non-blocking way: using signals and slots instead of directly changing
        the log text box
    """
    sigLog = QtCore.Signal(str)  # define the signal that will send the log messages

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
    plotter_cv2 = QtCore.Signal(np.ndarray, int, str)
    # expecting array to plot, int with the time to waitKey and a string with the label to show


    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        # connect plot signal to the plotting function. This way gui handles it when signal emision
        self.plotter_cv2.connect(self.show_cv2_image) #type=QtCore.Qt.BlockingQueuedConnection


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

        # Initialize a worker for the hard tasks
        self.strong_worker = Worker( None, None)
        self.strong_worker.finished.connect(lambda:
            self.block_hard_user_interaction(True))

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

    def initialize_Angle_Calculator_instance_convert_images(self):
        """
            Initializes an instance of the Image_Loader class and executes the
            conversion of the raw images to the selected mode i607 or i203. The output images
            will be saved in the output directory.

        """
        # Block everything to user
        self.block_hard_user_interaction(False)

        # save the mode chosen by the user
        self.mode = 607 if self.use_i607.isChecked() else 203
        # initialize instance
        self.image_loader = Image_Loader(mode=self.mode)

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
        # Get arguments and run algorithm depending on the chosen stuff



    def _check_image_loader_ready(self):
        """
        Checks whether images have already been converted and whether user wants them to
        be used or rather to import already converted images. To be used in secondary
        thread before the execution of any algorithm. Algorithms from secondary threads
        will call it.
        """
        if self.use_current_images_rot.isChecked():
            # make sure there is an instance of image_loader initilized!
            if not self.image_loader_initialized:
                self.mode = 607 if self.use_i607.isChecked() else 203
                # initialize instance
                self.image_loader = Image_Loader(mode=self.mode)
                ret = self._initialize_Angle_Calculator_instance_convert_images()
                return 1 if ret==1
        else: # use all the images in the output directory
            self.mode = 607 if self.use_converted_i607.isChecked() else 203
            self.image_loader = Image_Loader(mode=self.mode)
            ret = self.image_loader.import_converted_images(
                sorted(glob(f"{self.output_directory}/i{self.mode}_converted_images/*")))
            return 1 if ret==1












if __name__ == "__main__":

    # Initialize and execute app
    app = QtWidgets.QApplication([])
    window = Polarization_by_Conical_Refraction()
    window.show()
    app.exec_()
