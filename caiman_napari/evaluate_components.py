from PyQt5 import QtWidgets, QtCore
from .eval_components_template import Ui_Form


class EvalComponentsWidgets(QtWidgets.QWidget):
    sig_param_changed = QtCore.pyqtSignal()

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        for obj in self.ui.__dict__.keys():
            if obj.startswith('doubleSpinBox_'):
                getattr(self.ui, obj).valueChanged.connect(self.sig_param_changed)

        self.sig_param_changed.connect(self.update_components)

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
