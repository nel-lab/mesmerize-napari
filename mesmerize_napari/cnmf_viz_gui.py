from PyQt5 import QtWidgets
from .viz_pytemplate import Ui_VizualizationWidget
from .evaluate_components import EvalComponentsWidgets
from mesmerize_napari.core.utils import *
from .core import *
import caiman as cm



class VizWidget(QtWidgets.QDockWidget):
    def __init__(self, cnmf_viewer, batch_item):
        QtWidgets.QDockWidget.__init__(self, parent=None)
        self.ui = Ui_VizualizationWidget()
        self.ui.setupUi(self)
        self.batch_item = batch_item
        self.cnmf_viewer = cnmf_viewer
        self.eval_gui = EvalComponentsWidgets(cnmf_viewer=cnmf_viewer)

        self.ui.pushButtonInputMovie.clicked.connect(self.view_input)
        self.ui.pushButtonCnImage.clicked.connect(self.load_correlation_image)
        self.ui.pushButtonViewProjection.clicked.connect(self.view_projections)
        self.ui.pushButtonEvalGui.clicked.connect(self.show_eval_gui)
    def _open_movie(self, path: Union[Path, str]):
        file_ext = pathlib.Path(path).suffix
        if file_ext == '.mmap':
            Yr, dims, T = cm.load_memmap(path)
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            self.cnmf_viewer.viewer.add_image(images, colormap='gray')
        else:
            self.cnmf_viewer.viewer.open(path, colormap='gray')

    def view_input(self):
        path = self.batch_item.caiman.get_input_movie_path()
        full_path = get_full_data_path(path)
        self._open_movie(full_path)

    def load_correlation_image(self):
        corr_img = self.batch_item.caiman.get_correlation_image()
        self. cnmf_viewer.viewer.add_image(corr_img, name=f'corr: {self.batch_item["name"]}', colormap='gray')
    def view_projections(self):
        proj_type = self.ui.comboBoxProjection.currentText()
        projection = self.batch_item.caiman.get_projection(proj_type=proj_type)
        self.cnmf_viewer.viewer.add_image(projection, name=f'{proj_type} projection: {self.batch_item["name"]}', colormap='gray')

    def show_eval_gui(self):
        self.eval_gui.show()
