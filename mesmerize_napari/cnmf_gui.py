from PyQt5 import QtWidgets
from .cnmf_pytemplate import Ui_CNMFParmsWindow
from mesmerize_core.utils import *


class CNMFWidget(QtWidgets.QMainWindow):
    def __init__(self, parent):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        self.ui = Ui_CNMFParmsWindow()
        self.ui.setupUi(self)
        self.ui.btnAddToBatchCNMF.clicked.connect(self.add_item)

    @present_exceptions()
    def get_params(self, *args, group_params: bool = True) -> Tuple[str, bool, dict]:
        """
        Get a dict of the set parameters.
        If the work environment was loaded from a motion correction batch item it put the bord_px in the dict.
        Doesn't use any arguments

        :return: parameters dict
        :rtype: dict
        """
        # TODO: Figure out how to get framerate & bord_pix

        rf = self.ui.spinBoxRf.value()

        # CNMF kwargs
        cnmf_kwargs = \
            {
                'p': self.ui.spinBoxP.value(),
                'nb': self.ui.spinBoxnb.value(),
                # raises error: no parameter 'merge_thresh' found
                #'merge_thresh': self.ui.doubleSpinBoxMergeThresh.value(),
                'rf': rf if not self.ui.checkBoxRfNone.isChecked() else None,
                'stride': self.ui.spinBoxStrideCNMF.value(),
                'K': self.ui.spinBoxK.value(),
                'gSig': [
                    self.ui.spinBox_gSig_x.value(),
                    self.ui.spinBox_gSig_y.value()
                ],
                'ssub': self.ui.spinBox_ssub.value(),
                'tsub': self.ui.spinBox_tsub.value(),
                'method_init': self.ui.comboBox_method_init.currentText(),
                # 'border_pix': bord_px,
                # 'fr': self.vi.viewer.workEnv.imgdata.meta['fps']
            }

        # Any additional cnmf kwargs set in the text entry
        if self.ui.groupBox_cnmf_kwargs.isChecked():
            try:
                _kwargs = self.ui.plainTextEdit_cnmf_kwargs.toPlainText()
                cnmf_kwargs_add = eval(f"dict({_kwargs})")
                cnmf_kwargs.update(cnmf_kwargs_add)
            except:
                raise ValueError("CNMF kwargs not formatted properly.")

        # Component evaluation kwargs
        eval_kwargs = \
            {
                'min_SNR': self.ui.doubleSpinBoxMinSNR.value(),
                'rval_thr': self.ui.doubleSpinBoxRvalThr.value(),
                'use_cnn': self.ui.checkBoxUseCNN.isChecked(),
                'min_cnn_thr': self.ui.doubleSpinBoxCNNThr.value(),
                'cnn_lowest': self.ui.doubleSpinBox_cnn_lowest.value(),
                'decay_time': self.ui.spinBoxDecayTime.value(),
                # 'fr': self.vi.viewer.workEnv.imgdata.meta['fps']
            }

        # Any additional eval kwargs set in the text entry
        if self.ui.groupBox_eval_kwargs.isChecked():
            try:
                _kwargs = self.ui.plainTextEdit_eval_kwargs.toPlainText()
                eval_kwargs_add = eval(f"dict{_kwargs}")
                eval_kwargs.update(eval_kwargs_add)
            except:
                raise ValueError("Evaluation kwargs not formatted properly.")

        # Make the output dict
        d = dict()
        d.update(
            {
                'refit': self.ui.checkBoxRefit.isChecked()
            }
        )

        # Group the kwargs of the two parts seperately
        if group_params:
            d.update(
                {
                    'cnmf_kwargs': cnmf_kwargs,
                    'eval_kwargs': eval_kwargs
                }
            )

        # or not
        else:
            d.update(
                {
                    **cnmf_kwargs,
                    **eval_kwargs
                }
            )

        name = self.ui.lineEdName.text()
        refit = self.ui.checkBoxRefit.isChecked()
        print("cnmf_gui get params:", d)

        return name, d

    def add_item(self):
        item_name, params = self.get_params()
        print("cnmf_gui add params:", params)

        self.parent().add_item(algo='cnmf', parameters=params, name=item_name)

