from PyQt5 import QtWidgets
from .mcorr_pytemplate import Ui_MCORRWidget
from .utils import *
from .core import *
from napari import Viewer


class MCORRWidget(QtWidgets.QDockWidget):
    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, parent=parent)
        self.ui = Ui_MCORRWidget()
        self.ui.setupUi(self)
        self.ui.btnAddToBatchElastic.clicked.connect(self.add_item)

    #@present_exceptions()
    def get_params(self, *args, group_params: bool = True) -> Tuple[str, dict]:
        """
        Get a dict of the set parameters.
        If the work environment was loaded from a motion correction batch item it put the bord_px in the dict.
        Doesn't use any arguments

        :return: parameters dict
        :rtype: dict
        """
        # TODO: Change mcorr gui to include different parameters
        gSig = self.ui.spinBoxGSig_filt.value()
        if gSig == 0:
            gSig = None
        else:
            gSig = (gSig, gSig)

        mcorr_kwargs = \
            {
                'max_shifts': [self.ui.spinboxX.value(), self.ui.spinboxY.value()],
                'strides': [self.ui.spinboxStrides.value(), self.ui.spinboxStrides.value()],
                'overlaps': (self.ui.spinboxOverlaps.value(), self.ui.spinboxOverlaps.value()),
                'max_deviation_rigid': self.ui.spinboxMaxDev.value(),
                'border_nan': 'copy',
                'pw_rigid': self.ui.checkBoxRigidMC.isChecked(),
                'gSig_filt': gSig
            }
        # Any additional mcorr kwargs set in the text entry
        if self.ui.groupBox_motion_correction_kwargs.isChecked():
            try:
                _kwargs = self.ui.plainTextEdit_mc_kwargs.toPlainText()
                mcorr_kwargs_add = eval(f"dict({_kwargs})")
                mcorr_kwargs.update(mcorr_kwargs_add)
            except:
                raise ValueError("MCorr kwargs not formatted properly.")

        # Make the output dict
        d = dict()

        # Group the kwargs of the two parts separately
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
                    **mcorr_kwargs
                }
            )

        name = self.ui.lineEditNameElastic.text()

        return name, d

    def add_item(self):
        item_name, params = self.get_params()
        print("mcorr params", params)

        self.parent().add_item(algo='mcorr', parameters=params, name=item_name)

