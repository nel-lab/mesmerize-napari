from PyQt5 import QtWidgets
from .mcorr_pytemplate import Ui_DockWidget
from .utils import *
from .core import *
from napari import Viewer


class MCORRWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.ui = Ui_DockWidget()
        self.ui.setupUi(self)

    @present_exceptions()
    def get_params(self, *args, group_params: bool = False) -> dict:
        """
        Get a dict of the set parameters.
        If the work environment was loaded from a motion correction batch item it put the bord_px in the dict.
        Doesn't use any arguments

        :return: parameters dict
        :rtype: dict
        """
        # TODO: Change mcorr gui to include different parameters

        mcorr_kwargs = \
            {
                'max_shifts': [self.ui.spinboxX.value(), self.ui.spinboxY.value()],
                'strides': [self.ui.spinboxStrides.value(), self.ui.spinboxStrides.value()],
                'overlaps': (self.ui.spinboxOverlaps.value(), self.ui.spinboxOverlaps.value()),
                'max_deviation_rigid': self.ui.spinboxMaxDev.value(),
                'border_nan': 'copy',
                'decay_time': 0.4,
                'pw_rigid': True
            }
        # Any additional mcorr kwargs set in the text entry
        if self.ui.groupBox_motion_correction_kwargs.isChecked():
            try:
                _kwargs = self.ui.plainTextEdit_mc_kwargs.toPlainText()
                mcorr_kwargs_add = eval(f"dict({_kwargs})")
                mcorr_kwargs.update(mcorr_kwargs_add)
            except:
                raise ValueError("CNMF kwargs not formatted properly.")

        if self.vi.viewer.workEnv.imgdata.ndim == 4:
            is_3d = True
        else:
            is_3d = False

        # Make the output dict
        d = \
            {
                'item_name': self.ui.lineEdName.text(),
                'is_3d': is_3d,
                'keep_memmap': self.ui.checkBoxKeepMemmap.isChecked()
            }
        # Group the kwargs of the two parts seperately
        if group_params:
            d.update(
                {
                    'mcorr_kwargs': mcorr_kwargs
                }
            )

        # or not
        else:
            d.update(
                {
                    **mcorr_kwargs,
                }
            )

        return d

    def add_item(self):
        params = self.get_params()
        item_name = self.ui.lineEdName.text()

        self.parent().add_item(algo='mcorr', parameters=params, name=item_name)

