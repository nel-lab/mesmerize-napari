from PyQt5 import QtWidgets
from .cnmfe_pytemplate import Ui_CNMFEDockWidget
from .utils import *
from .core import *

class CNMFEWidget(QtWidgets.QDockWidget):
    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, parent=parent)
        self.ui = Ui_CNMFEDockWidget()
        self.ui.setupUi(self)
        self.ui.btnAddToBatchCorrPNR.clicked.connect(self._add_item_corr_pnr)
        self.ui.btnAddToBatchCNMFE.clicked.connect(self._add_item_cnmfe)

    @present_exceptions()
    def _add_item_corr_pnr(self, *args) -> Tuple[str, dict]:
        d = dict()
        d.update(
            {
                'do_cnmfe': False,
                'gSig': self.ui.spinBoxGSig.value(),
            }
        )

        name = self.ui.lineEdCorrPNRName.text()

        self.add_item(name, d)

    @present_exceptions()
    def _add_item_cnmfe(self, *args, group_params: bool = True) -> Tuple[str, bool, dict]:
        low_rank_background = self.ui.checkBox_low_rank_background.isChecked()
        keep_memmap = self.ui.checkBoxKeepMemmap.isChecked()
        # CNMF kwargs
        cnmfe_kwargs = \
            {
                'gSig': self.ui.spinBoxGSig.value(),
                'p': self.ui.spinBox_p.value(),
                'min_corr': self.ui.doubleSpinBoxMinCorr.value(),
                'min_pnr': self.ui.spinBoxMinPNR.value(),
                'rf': self.ui.spinBoxRf.value(),
                'overlap': self.ui.spinBoxOverlap.value(),
                'gnb': self.ui.spinBoxGnb.value(),
                'nb_patch': self.ui.spinBoxNb_patch.value(),
                'k': self.ui.spinBoxK.value(),
                'ssub': self.ui.spinBox_ssub.value(),
                'tsub': self.ui.spinBox_tsub.value(),
                'ring_size_factor': self.ui.doubleSpinBox_ring_size_factor.value(),
                'deconvolution': self.ui.comboBoxDeconv.currentText(),
                'merge_thresh': self.ui.doubleSpinBoxMergeThresh.value(),
            }
        # Any additional cnmfe kwargs set in the text entry
        if self.ui.groupBox_cnmf_kwargs.isChecked():
            try:
                _kwargs = self.ui.plainTextEdit_cnmf_kwargs.toPlainText()
                cnmfe_kwargs_add = eval(f"dict({_kwargs})")
                cnmfe_kwargs.update(cnmfe_kwargs_add)
            except:
                raise ValueError("CNMF-E kwargs not formatted properly.")

        # Make the output dict
        d = dict()
        d.update(
            {
                'do_cnmfe': True
            }
        )

        # Group the kwargs of the two parts separately
        if group_params:
            d.update(
                {
                    'cnmfe_kwargs': cnmfe_kwargs,
                    'low_rank_background': low_rank_background,
                    'keep_memmap': keep_memmap,
                }
            )
        else:
            d.update(
                {
                    **cnmfe_kwargs,
                    **low_rank_background,
                    **keep_memmap,
                }
            )
        name = self.ui.lineEdName.text()

        self.add_item(name, d)

    def add_item(self, item_name: str, params: dict):
        self.parent().add_item(algo='cnmfe', parameters=params, name=item_name)
