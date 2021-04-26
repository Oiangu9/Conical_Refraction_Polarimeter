from GUI.Design_ui import *
from SOURCE.Polarization_Angle_Calculator import *
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


class QPlainTextEditLogger_NonBlockong(logging.Handler, QtCore.QObject):
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
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args) # si pones *self.args se desacoplan todos los argumentos


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)


        # set the working directories
        self.master_directory = os.getcwd()
        self.image_directory.setText(self.master_directory+'/DATA/')
        self.output_directory.setText(self.master_directory+'/OUTPUT/')

        # set app icon
        self.setWindowIcon(QtGui.QIcon(self.master_directory+'/GUI/ICONS/croissant5.png'))

        # Initialize tree view of directories
        self.load_project_structure(self.master_directory+'/DATA/', self.picture_dir_tree)


        # Set up logging to use your widget as a handler
        log_handler = QPlainTextEditLogger_NonBlockong(self.log_text)
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

        # When the user clicks to choose the image or outpu directory a prompt will show up
        self.change_image_directory.clicked.connect(lambda:
            self.choose_directory("Choose directory for Images",  self.image_directory))
        self.change_output_directory.clicked.connect(lambda:
            self.choose_directory("choose directory for Output", self.output_directory))

        # when the button to convert selected images is pressed, run it
        self.image_loader_initialized = False # checking whether an instance for the Image_Loader has been already initialized
        self.convert_selected_images.clicked.connect(
            self.initialize_Angle_Calculator_instance_convert_images)



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

    def initialize_Angle_Calculator_instance_convert_images(self):
        """
            Initializes an instance of the Image_Loader class and executes the
            conversion of the raw images to the selected mode i607 or i203. The output images
            will be saved in the output directory.

        """
        # Block everything to user

        # Prepare thread for computations to allow non-blocking gui

        # save the mode chosen by the user
        self.mode = 607 if self.use_i607.isChecked() else 203
        # initialize instance
        self.image_loader = Image_Loader(mode=self.mode)
        # import the images
        self.image_loader.get_raw_images_to_compute(
            self.get_selected_file_paths(self.picture_dir_tree, self.image_directory.text()))
        # create directory for outputing the resulting converted images
        os.makedirs(self.output_directory.text()+f"/i{self.mode}_converted_images/", exist_ok=True)
        # convert the images
        self.image_loader.compute_raw_to_i607_or_i203(self.output_directory.text()+
                                                    f"/i{self.mode}_converted_images/")

        self.image_loader_initialized=True

        # Unblock things in gui

        








if __name__ == "__main__":

    # Initialize and execute app
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
