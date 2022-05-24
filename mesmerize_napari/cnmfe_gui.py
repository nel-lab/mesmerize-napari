from PyQt5 import QtWidgets
from .cnmfe_pytemplate import Ui_CNMFEParamsWindow
from mesmerize_core.utils import *


class CNMFEWidget(QtWidgets.QMainWindow):
    def __init__(self, parent):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        self.ui = Ui_CNMFEParamsWindow()
        self.ui.setupUi(self)
        self.ui.btnAddToBatchCorrPNR.clicked.connect(self._add_item_corr_pnr)
        self.ui.btnAddToBatchCNMFE.clicked.connect(self._add_item_cnmfe)

    @present_exceptions()
    def _add_item_corr_pnr(self, *args):
        gSig = self.ui.spinBoxGSig.value()
        downsample_ratio = self.ui.spinBoxDownsample.value()
        cnmfe_kwargs = {"gSig": (gSig, gSig)}
        d = dict()
        d.update(
            {
                "do_cnmfe": False,
                "cnmfe_kwargs": cnmfe_kwargs,
                "downsample_ratio": downsample_ratio,
            }
        )

        name = self.ui.lineEdCorrPNRName.text()

        self.add_item(name, d)

    @present_exceptions()
    def _add_item_cnmfe(self, *args, group_params: bool = True):
        low_rank_background = self.ui.checkBox_low_rank_background.isChecked()
        # CNMF kwargs
        gSig = self.ui.spinBoxGSig.value()
        kval = self.ui.spinBoxK.value()
        if kval == 0:
            kval = None
        deconv = self.ui.comboBoxDeconv.currentText()
        if deconv == "SKIP":
            method_deconvolution = None
        else:
            method_deconvolution = deconv
        downsample_ratio = self.ui.spinBoxDownsample.value()
        cnmfe_kwargs = {
            "gSig": (gSig, gSig),
            "gSiz": (4 * gSig + 1, 4 * gSig + 1),
            "p": self.ui.spinBox_p.value(),
            "min_corr": self.ui.doubleSpinBoxMinCorr.value(),
            "min_pnr": self.ui.spinBoxMinPNR.value(),
            "rf": self.ui.spinBoxRf.value(),
            "stride": self.ui.spinBoxOverlap.value(),
            "gnb": self.ui.spinBoxGnb.value(),
            "nb_patch": self.ui.spinBoxNb_patch.value(),
            "K": kval,
            "ssub": self.ui.spinBox_ssub.value(),
            "tsub": self.ui.spinBox_tsub.value(),
            "ring_size_factor": self.ui.doubleSpinBox_ring_size_factor.value(),
            "merge_thresh": self.ui.doubleSpinBoxMergeThresh.value(),
            "low_rank_background": low_rank_background,
            "method_deconvolution": method_deconvolution,
            "update_background_components": True,
            "del_duplicates": True,
            "fr": self.ui.doubleSpinBoxFrameRate.value(),
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
                "do_cnmfe": True,
                "downsample_ratio": downsample_ratio,
            }
        )

        # Group the kwargs of the two parts separately
        if group_params:
            d.update(
                {
                    'cnmfe_kwargs': cnmfe_kwargs,
                }
            )
        else:
            d.update(
                {
                    **cnmfe_kwargs,
                }
            )
        name = self.ui.lineEdName.text()

        self.add_item(item_name=name, params=d)

    def add_item(self, item_name: str, params: dict):
        print("added cnmfe item", params)
        self.parent().add_item(algo="cnmfe", parameters=params, name=item_name)
