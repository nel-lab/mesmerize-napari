from PyQt5 import QtWidgets
from .mcorr_viz_pytemplate import Ui_VizualizationWidget
import caiman as cm
from mesmerize_core import *
from mesmerize_core.utils import *
from mesmerize_core.batch_utils import *
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
        self.ui.pushButtonViewDownsampleMCMovie.clicked.connect(self.view_downsample_mcorr)

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
        full_path = get_full_raw_data_path(path)
        self._open_movie(full_path)

    def load_correlation_image(self):
        corr_img = self.batch_item.caiman.get_corr_image()
        self.mcorr_viewer.viewer.add_image(
            corr_img, name=f'corr: {self.batch_item["item_name"]}', colormap="gray"
        )

    def view_projections(self):
        proj_type = self.ui.comboBoxProjection.currentText()
        projection = self.batch_item.caiman.get_projection(proj_type=proj_type)
        self.mcorr_viewer.viewer.add_image(
            projection,
            name=f'{proj_type} projection: {self.batch_item["item_name"]}',
            colormap="gray",
        )

    def view_downsample_mcorr(self):
        downsample_window = self.ui.spinBoxDownsampleWindow.value()
        self.ds_video = self.batch_item.mcorr.get_output()
        frame0 = np.nanmean(self.ds_video[0:downsample_window], axis=0)
        self.mcorr_viewer.viewer.add_image(
                    frame0,
                    name='Downsampled MC Movie')
        self.mcorr_viewer.viewer.dims.events.current_step.connect(self.update_slider)

    def update_slider(self, event):
        downsample_window = self.ui.spinBoxDownsampleWindow.value()
        ix = self.mcorr_viewer.viewer.dims.current_step[0]
        start = max(0, ix - downsample_window)
        end = min(self.ds_video.shape[0], ix + downsample_window)
        ds_frame = np.nanmean(self.ds_video[start:end], axis=0)
        self.mcorr_viewer.viewer.layers['Downsampled MC Movie'].data = ds_frame