from qtpy import QtWidgets
from napari_plugin_engine import napari_hook_implementation
from napari import Viewer
from napari.layers import Layer, Shapes
from .common import *
from pathlib import Path

import numpy as np
from time import time
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import json
from tqdm import tqdm
from caiman.utils.visualization import get_contours as caiman_get_contours


class CNMF(QtWidgets.QWidget):
    def __init__(self, napari_viewer: Viewer):
        self.viewer: Viewer = napari_viewer
        QtWidgets.QWidget.__init__(self)

        vlayout = QtWidgets.QVBoxLayout()

        btn_open = QtWidgets.QPushButton('Open movie', self)
        btn_open.clicked.connect(self._open_image_dialog)
        vlayout.addWidget(btn_open)

        # just a button to start CNMF with some hard coded params for this prototype
        btn_start_cnmf = QtWidgets.QPushButton('Perform CNMF', self)
        btn_start_cnmf.clicked.connect(self.start_cnmf)

        # activate layout
        self.setLayout(vlayout)

        self.path: str = None
        self.cnmf_obj: cnmf.CNMF = None

    @use_open_file_dialog('Choose image file', '', ['*.tiff', '*.tif', '*.btf'])
    def _open_image_dialog(self, path: str, *args, **kwargs):
        if not self.clear_viewer():
            return

        self.open_image(path)

    def open_image(self, path: str):
        self.path = path
        self.viewer.open(path)

    def clear_viewer(self) -> bool:
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

    def start_cnmf(self):
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local',
            n_processes=(psutil.cpu_count() - 1),
            single_thread=False, ignore_preexisting
            =True
        )

        memmap_fname = cm.save_memmap(
            filenames=[self.path],
            base_name=f'memmap-{os.path.basename(self.path)}',
            order='C',
            dview=dview
        )

        Yr, dims, T = cm.load_memmap(memmap_fname)
        Y = np.reshape(Yr.T, [T] + list(dims), order='F')

        # just some hard coded params for the prototype
        params = CNMFParams(
            params_dict=json.load(open('./params.json', 'r'))
        )

        self.cnmf_obj = cnmf.CNMF(
            dview=dview,
            n_processes=n_processes,
            params=params,
        )

        self.cnmf_obj.fit(Y)

        self.cnmf_obj.estimates.evaluate_components(
            Y,
            self.cnmf_obj.params,
            dview=dview
        )

        self.cnmf_obj.estimates.select_components(use_object=True)

        out_filename = f'{self.path}_results.hdf5'
        self.cnmf_obj.save(out_filename)

    def show_results(self):
        dims = self.cnmf_obj.dims
        if dims is None:  # I think that one of these is `None` if loaded from an hdf5 file
            dims = self.cnmf_obj.estimates.dims

        contours_good = caiman_get_contours(
            self.cnmf_obj.estimates.A[:, self.cnmf_obj.estimates.idx_components],
            dims
        )

        colors_contours_good = auto_colormap(
            n_colors=len(contours_good),
            cmap='hsv',
            output='mpl',
        )

        contours_bad = caiman_get_contours(
            self.cnmf_obj.estimates.A[:, self.cnmf_obj.estimates.idx_components_bad],
            dims
        )

        for i in tqdm(range(len(contours_good)), desc="Adding good components..."):
            pass

        for i in tqdm(range(len(contours_bad)), desc="Adding bad components..."):
            pass


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return CNMF
