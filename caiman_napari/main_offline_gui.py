import time
from .main_offline_gui_template import Ui_MainOfflineGUIWidget
from .mcorr_gui import MCORRWidget
from .cnmf_gui import CNMFWidget
from PyQt5 import QtWidgets
from qtpy import QtWidgets
from napari_plugin_engine import napari_hook_implementation
from napari import Viewer
from .utils import *
from .core import *
import pandas as pd
from functools import partial
import pprint
from . import algorithms
import caiman as cm


COLORS_HEX = \
    {
        'orange': '#ffb347',
        'green': '#77dd77',
        'red': '#fe0d00',
        'blue': '#85e3ff'
    }


class MainOfflineGUI(QtWidgets.QWidget):
    def __init__(self, napari_viewer: Viewer):
        self.viewer = napari_viewer
        QtWidgets.QWidget.__init__(self)

        self.ui = Ui_MainOfflineGUIWidget()
        self.ui.setupUi(self)
        self.show()

        self.input_movie_path = None

        self.dataframe: pd.DataFrame = None
        self.dataframe_file_path: str = None
        # define actions for each button
        ## Open Movie
        self.ui.pushButtonOpenMovie.clicked.connect(self.open_movie)
        ## Open Panel to set parameters for CNMF
        self.ui.pushButtonParamsCNMF.clicked.connect(self.show_cnmf_params_gui)
        ## Open panel for MCORR
        self.ui.pushButtonParamsMCorr.clicked.connect(self.show_mcorr_params_gui)
        ## Start Batch
        self.ui.pushButtonNewBatch.clicked.connect(self.create_new_batch)
        ## Open Batch
        self.ui.pushButtonOpenBatch.clicked.connect(self.open_batch)
        ## Start running from zereoth index
        self.ui.pushButtonStart.clicked.connect(self.run)
        ## Start running from selected index
        self.ui.pushButtonStartItem.clicked.connect(self.run_item)
        ## Remove selected item
        self.ui.pushButtonDelItem.clicked.connect(self.remove_item)
        ## Change parameters displayed in param text box upon change in selection
        self.ui.listWidgetItems.currentRowChanged.connect(self.set_params_text)
        ## On double click of item in listwidget, load results and display
        self.ui.listWidgetItems.doubleClicked.connect(self.load_output)

    @use_open_file_dialog('Choose image file', '', ['*.tiff', '*.tif', '*.btf', '*.mmap'])
    def open_movie(self, path: str, *args, **kwargs):
        if not self.clear_viewer():
            return

        self.input_movie_path = path
        file_ext = pathlib.Path(self.input_movie_path).suffix
        if file_ext == '.mmap':
            Yr, dims, T = cm.load_memmap(path)
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            self.viewer.add_image(images)
        else:
            self.viewer.open(self.input_movie_path)



    def clear_viewer(self) -> bool:
        if len(self.viewer.layers) == 0:
            return True

        if QtWidgets.QMessageBox.warning(
                self,
                'Clear viewer?',
                'The viewer must be cleared, do you want to continue?',
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        ) == QtWidgets.QMessageBox.No:
            return False

        for layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    @use_save_file_dialog('Choose location to save batch file', '', '.pickle')
    def create_new_batch(self, path, *args, **kwargs):
        self.ui.listWidgetItems.clear()
        self.dataframe = create_batch(path)
        self.dataframe_file_path = path

    @use_open_file_dialog('Choose batch', '', ['*.pickle'])
    def open_batch(self, path: str, *args, **kwargs):
        self.dataframe = load_batch(path)
        self.dataframe_file_path = path

        self.ui.listWidgetItems.clear()

        # Iterate through dataframe, add each item to list widget
        ## For now, instead of name I'm adding the uuid
        for i, r in self.dataframe.iterrows():
            algo = r['algo']
            name = r['name']
            uuid = r['uuid']
            self.ui.listWidgetItems.addItem(f'{algo}: {name}')

            item = self.ui.listWidgetItems.item(i)
            item.setData(3, uuid)

    def add_item(self, algo: str, parameters: dict, name: str, input_movie_path: str = None):
        if input_movie_path is None:
            input_movie_path = self.input_movie_path

        self.dataframe.caiman.add_item(
            algo=algo, name=name, input_movie_path=input_movie_path, params=parameters
        )
        print("after add_item", self.dataframe.params)

        uuid = self.dataframe.iloc[-1]['uuid']

        self.ui.listWidgetItems.addItem(f'{algo}: {name}')

        n = self.ui.listWidgetItems.count()
        item = self.ui.listWidgetItems.item(n - 1)
        item.setData(3, uuid)

    def remove_item(self):
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)

        ix = self.dataframe[self.dataframe['uuid'] == uuid].index[0]
        self.dataframe.caiman.remove_item(index=ix)
        self.ui.listWidgetItems.takeItem(ix)

    def run_item(self):
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)

        ix = self.dataframe[self.dataframe['uuid'] == uuid].index[0]

        # For now, just run given index.
        self._run_index(ix)

    def run(self):
        self._run_index(0)

    def _run_index(self, index: int):
        callbacks = [partial(self.item_finished, index)]
        std_out = self._print_qprocess_std_out

        self.dataframe.iloc[index].caiman.run(callbacks_finished=callbacks, callback_std_out=std_out)

    def _print_qprocess_std_out(self, proc):
        txt = proc.readAllStandardOutput().data().decode('utf8')
        self.ui.textBrowserStdOut.append(txt)

    def item_finished(self, ix):
        self.dataframe = load_batch(self.dataframe_file_path)
        if self.dataframe.iloc[ix]['outputs']['success']:
            self.set_list_widget_item_color(ix, 'green')
        else:
            self.set_list_widget_item_color(ix, 'red')

        if (ix + 1) < self.ui.listWidgetItems.count():
            time.sleep(10)
            self._run_index(ix + 1)

        else:
            QtWidgets.QMessageBox.information(self, 'Batch is done!', 'Yay, your batch has finished processing!')

    def set_params_text(self, ix):
        p = self.dataframe.iloc[ix]['params']
        print(p)
        u = self.dataframe.iloc[ix]['uuid']
        s = pprint.pformat(p)
        s = f"{u}\n\n{s}"

        if self.dataframe.iloc[ix]['outputs'] is not None:
            if self.dataframe.iloc[ix]['outputs']['traceback'] is not None:
                tb = self.dataframe.iloc[ix]['outputs']['traceback']
                s += f"\n\n{tb}"

        self.ui.textBrowserParams.setText(s)

    def set_list_widget_item_color(self, ix: int, color: str):
        self.ui.listWidgetItems.item(ix).setBackground(QtGui.QBrush(QtGui.QColor(COLORS_HEX[color])))

    def show_cnmf_params_gui(self):
        self.cnmf_gui = CNMFWidget(parent=self)
        self.cnmf_gui.show()

    def show_mcorr_params_gui(self):
        self.mcorr_gui = MCORRWidget(parent=self)
        self.mcorr_gui.show()

    def load_output(self):
        # clear napari viewer before loading new movies
        self.clear_viewer()
        # Find uuid for selected item
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)
        # Algorithm name for selected item
        algo = self.dataframe.loc[self.dataframe['uuid'] == uuid, 'algo'].item()
        # Open input movie for selected item
        self.viewer.open(self.dataframe.loc[self.dataframe['uuid'] == uuid, 'input_movie_path'].item())
        print("show outputs for: ", algo)

        getattr(algorithms, algo).load_output(self.viewer, self.dataframe.loc[self.dataframe['uuid'] == uuid])


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MainOfflineGUI
