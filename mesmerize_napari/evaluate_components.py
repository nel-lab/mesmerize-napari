from PyQt5 import QtWidgets, QtCore
from .eval_components_template import Ui_EvalComponents

from mesmerize_core import CNMFExtensions
from caiman.source_extraction.cnmf.cnmf import CNMFParams, CNMF


class EvalComponentsWidgets(QtWidgets.QMainWindow):
    sig_param_changed = QtCore.pyqtSignal()

    def __init__(self, cnmf_viewer):
        QtWidgets.QMainWindow.__init__(self, parent=None)
        self.ui = Ui_EvalComponents()
        self.ui.setupUi(self)

        self.cnmf_viewer = cnmf_viewer

        for obj in self.ui.__dict__.keys():
            if obj.startswith('doubleSpinBox_'):
                getattr(self.ui, obj).valueChanged.connect(self.sig_param_changed)

        if self.ui.checkBox_update_live.isChecked():
            self.sig_param_changed.connect(self.update_components)

        self.ui.pushButton_update.clicked.connect(self.update_components)
        self.ui.checkBox_update_live.toggled.connect(self.set_live_update)

    def set_live_update(self, b: bool):
        if b:
            self.sig_param_changed.connect(self.update_components)
        else:
            self.sig_param_changed.disconnect()

    def get_params(self):
        d = dict()
        for obj in self.ui.__dict__.keys():
            if obj.startswith('doubleSpinBox_'):
                param = obj.split('doubleSpinBox_')[1]
                val = getattr(self.ui, obj).value()
                d[param] = val

        return d

    def update_components(self):
        params = self.get_params()
        cnmf_obj: CNMF = self.cnmf_viewer.cnmf_obj
        cnmf_obj.params.quality.update(params)

        cnmf_obj.estimates.filter_components(
            imgs=self.cnmf_viewer.batch_item.cnmf.get_input_memmap(),
            params=cnmf_obj.params,
        )

        self.cnmf_viewer.update_visible_components()
