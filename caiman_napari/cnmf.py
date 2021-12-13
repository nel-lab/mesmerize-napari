import os.path

import napari.viewer
from qtpy import QtWidgets, QtCore
from napari_plugin_engine import napari_hook_implementation
from napari import Viewer
from napari.layers import Layer, Shapes
from .common import *
from pathlib import Path
from napari.utils import io
# TODO: ask where frame rate metadata stored in napari

import numpy as np
from time import time
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import psutil
import json
from tqdm import tqdm
from caiman.utils.visualization import get_contours as caiman_get_contours
from functools import partial
from . import _cnmf
from . import _mcorr
from pyqtgraph import PlotDataItem, PlotWidget
from .main_dockwidget import Ui_DockWidget


class CNMF(QtWidgets.QWidget):
    def __init__(self, napari_viewer: Viewer):
        self.viewer: Viewer = napari_viewer
        QtWidgets.QWidget.__init__(self)
        self.ui = Ui_DockWidget()
        self.ui.setupUi(self)

        vlayout = QtWidgets.QVBoxLayout()

        self.ui.pushButtonOpenMovie.clicked.connect(self._open_image_dialog)

        # Button to run MC
        btn_params = QtWidgets
        btn_mc = QtWidgets.QPushButton('Run Motion Correction', self)
        btn_mc.clicked.connect(self.start_mc)
        vlayout.addWidget(btn_mc)

        # just a button to start CNMF with some hard coded params for this prototype
        btn_start_cnmf = QtWidgets.QPushButton('Perform CNMF', self)
        btn_start_cnmf.clicked.connect(self.start_cnmf)
        vlayout.addWidget(btn_start_cnmf)

        self.text_browser = QtWidgets.QTextBrowser(self)
        vlayout.addWidget(self.text_browser)

        # activate layout
        self.setLayout(vlayout)

        self.path: str = None
        self.cnmf_obj: cnmf.CNMF = None
        self.process: QtCore.QProcess = None
        self.plot_widget: PlotWidget = None

    @use_open_file_dialog('Choose image file', '', ['*.tiff', '*.tif', '*.btf'])
    def _open_image_dialog(self, path: str, *args, **kwargs):
        if not self.clear_viewer():
            return

        self.open_image(path)

    def open_image(self, path: str):
        self.path = path
        self.viewer.open(path)

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

        return True
    def start_mc(self):
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(partial(self._print_qprocess_std_out, self.process))
        self.process.finished.connect(self.show_mc_results)

        runfile = make_runfile(
            module_path=os.path.abspath(_mcorr.__file__),
            filename=self.path + '.runfile',
            args_str=self.path
        )

        self.process.setWorkingDirectory(os.path.dirname(self.path))

        if IS_WINDOWS:
            self.process.start('powershell.exe', [runfile])
        else:
            print(runfile)
            self.process.start(runfile)
    def show_mc_results(self):
        # extract mmap from _cnmf and display new movie
        mmap = np.load(self.path + 'mc.npy')
        vid = mmap[0]
        self.viewer.add_image(vid, name='Motion Corrected')

    def start_cnmf(self):
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(partial(self._print_qprocess_std_out, self.process))
        self.process.finished.connect(self.show_results)

        runfile = make_runfile(
            module_path=os.path.abspath(_cnmf.__file__),
            filename=self.path + '.runfile',
            args_str=self.path
        )

        self.process.setWorkingDirectory(os.path.dirname(self.path))

        if IS_WINDOWS:
            self.process.start('powershell.exe', [runfile])
        else:
            print(runfile)
            self.process.start(runfile)

    def _print_qprocess_std_out(self, proc):
        txt = proc.readAllStandardOutput().data().decode('utf8')
        self.text_browser.append(txt)

    def show_results(self):
        print("showing results")
        self.cnmf_obj = load_CNMF(self.path + '.results.hdf5')

        dims = self.cnmf_obj.dims
        if dims is None:  # I think that one of these is `None` if loaded from an hdf5 file
            dims = self.cnmf_obj.estimates.dims

        # need to transpose these
        dims = dims[1], dims[0]

        contours_good = caiman_get_contours(
            self.cnmf_obj.estimates.A[:, self.cnmf_obj.estimates.idx_components],
            dims,
            swap_dim=True
        )

        colors_contours_good = auto_colormap(
            n_colors=len(contours_good),
            cmap='hsv',
            output='mpl',
        )

        contours_good_coordinates = [self._organize_coordinates(c) for c in contours_good]
        self.viewer.add_shapes(
            data=contours_good_coordinates,
            shape_type='polygon',
            edge_width=0.5,
            edge_color=colors_contours_good,
            face_color=colors_contours_good,
            opacity=0.1,
        )

        if self.cnmf_obj.estimates.idx_components_bad is not None and len(self.cnmf_obj.estimates.idx_components_bad) > 0:
            contours_bad = caiman_get_contours(
                self.cnmf_obj.estimates.A[:, self.cnmf_obj.estimates.idx_components_bad],
                dims,
                swap_dim=True
            )

            contours_bad_coordinates = [self._organize_coordinates(c) for c in contours_bad]

            colors_contours_bad = auto_colormap(
                n_colors=len(contours_bad),
                cmap='hsv',
                output='mpl',
            )

            self.viewer.add_shapes(
                data=contours_bad_coordinates,
                shape_type='polygon',
                edge_width=0.5,
                edge_color=colors_contours_bad,
                face_color=colors_contours_bad,
                opacity=0.1,
            )

        self.plot_widget = PlotWidget()

        for c in self.cnmf_obj.estimates.C:
            self.plot_widget.addItem(PlotDataItem(c))

        self.plot_widget.show()

    def _organize_coordinates(self, contour: dict):
        coors = contour['coordinates']
        coors = coors[~np.isnan(coors).any(axis=1)]

        return coors


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return CNMF
