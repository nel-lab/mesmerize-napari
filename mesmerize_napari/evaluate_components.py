from PyQt5 import QtWidgets, QtCore
from .eval_components_template import Ui_MainWindow
from caiman.source_extraction.cnmf.cnmf import CNMFParams


class EvalComponentsWidgets(QtWidgets.QMainWindow):
    sig_param_changed = QtCore.pyqtSignal()

    def __init__(self, parent):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

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

        print(params)
