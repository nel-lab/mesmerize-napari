from .pytemplates.main_offline_gui_template import Ui_MainOfflineGUIWidget
from .cnmf_gui import CNMFWidget
from PyQt5 import QtWidgets, QtCore
from qtpy import QtWidgets, QtCore
from napari_plugin_engine import napari_hook_implementation
from napari import Viewer
from common import *
import pandas as pd
from functools import partial
from .core import *


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

        self.input_movie_path = None

        self.dataframe: pd.DataFrame = None
        self.dataframe_file_path: str = None

        self.cnmf_params_gui = CNMFWidget(parent=self)
        self.ui.pushButtonParamsCNMF.clicked.connect(self.cnmf_params_gui.show)

    @use_open_file_dialog('Choose image file', '', ['*.tiff', '*.tif', '*.btf'])
    def open_movie(self, path: str, *args, **kwargs):
        if not self.clear_viewer():
            return

        self.input_movie_path = path

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
        self.dataframe = create_batch(path)
        self.dataframe_file_path = path

    @use_open_file_dialog('Choose batch', '', ['*.pickle'])
    def open_batch(self, path: str, *args, **kwargs):
        self.dataframe = load_batch(path)
        self.dataframe_file_path = path

    def add_item(self, algo: str, parameters: dict, name: str, input_movie_path: str = None):
        if input_movie_path is None:
            input_movie_path = self.input_movie_path

        self.dataframe.caiman.add_item(algo, input_movie_path, parameters)
        uuid = self.dataframe.iloc[-1]['uuid']

        self.ui.listWidgetItems.addItem(f'{algo}: {name}')

        n = self.ui.listWidgetItems.count()
        item = self.ui.listWidgetItems.item(n - 1)
        item.setData(3, uuid)

    def run_item(self):
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)

        ix = self.dataframe[self.dataframe['uuid'] == uuid].index[0]

        self._run_index(ix)

    def run(self):
        self._run_index(0)

    def _run_index(self, index: int):
        callbacks = [partial(self.item_finished, index)]

        self.dataframe.iloc[index].caiman.run(callbacks_finished=callbacks)

    def item_finished(self, ix):
        self.set_list_widget_item_color(ix, 'green')

    def set_list_widget_item_color(self, ix: int, color: str):
        self.ui.listWidgetItems.item(ix).setBackground(QtGui.QBrush(QtGui.QColor(COLORS_HEX[color])))


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MainOfflineGUI
