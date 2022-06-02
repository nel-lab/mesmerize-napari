from PyQt5 import QtWidgets
from .mcorr_viz_pytemplate import Ui_VizualizationWidget
import caiman as cm
from mesmerize_core import *
from mesmerize_core.utils import *
import numpy as np

class MCORRVizWidget(QtWidgets.QDockWidget):
    def __init__(self, mcorr_viewer, batch_item):
        QtWidgets.QDockWidget.__init__(self, parent=None)
        self.ui = Ui_VizualizationWidget()
        self.ui.setupUi(self)
        self.mcorr_obj = batch_item.mcorr.get_output()
        self.batch_item = batch_item
        self.mcorr_viewer = mcorr_viewer

        self.ui.pushButtonInputMovie.clicked.connect(self.view_input)
        self.ui.pushButtonCnImage.clicked.connect(self.load_correlation_image)
        self.ui.pushButtonViewProjection.clicked.connect(self.view_projections)
        self.ui.pushButtonSubsampleMCMovie.clicked.connect(self.view_subsampled_movie)

    def _open_movie(self, path: Union[Path, str]):
        file_ext = Path(path).suffix
        if file_ext == ".mmap":
            Yr, dims, T = cm.load_memmap(path)
            images = np.reshape(Yr.T, [T] + list(dims), order="F")
            self.mcorr_viewer.viewer.add_image(images, colormap="gray")

        else:
            self.mcorr_viewer.viewer.open(path, colormap="gray")

    def view_input(self):
        path = self.batch_item.caiman.get_input_movie_path()
        full_path = get_full_data_path(path)
        self._open_movie(full_path)

    def load_correlation_image(self):
        corr_img = self.batch_item.caiman.get_correlation_image()
        self.mcorr_viewer.viewer.add_image(
            corr_img, name=f'corr: {self.batch_item["name"]}', colormap="gray"
        )

    def view_projections(self):
        proj_type = self.ui.comboBoxProjection.currentText()
        projection = self.batch_item.caiman.get_projection(proj_type=proj_type)
        self.mcorr_viewer.viewer.add_image(
            projection,
            name=f'{proj_type} projection: {self.batch_item["name"]}',
            colormap="gray",
        )
    def view_subsampled_movie(self):
        subsample_ratio = self.ui.spinBoxSubsampleRatio.value()
        images = self.batch_item.mcorr.get_output()[::subsample_ratio, :, :]
        self.mcorr_viewer.viewer.add_image(
            images,
            name = f"Subsampled MC Movie: {subsample_ratio}",
            colormap="gray",
        )
