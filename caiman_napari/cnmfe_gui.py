from PyQt5 import QtWidgets
from .cnmfe_pytemplate import Ui_CNMFEDockWidget
from .utils import *
from .core import *

class CNMFEWidget(QtWidgets.QDockWidget):
    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, parent=parent)
        self.ui = Ui_CNMFEDockWidget()
        self.ui.setupUi(self)

    @present_exceptions()
    def get_params(self, *args, group_params: bool = True) -> Tuple[str, bool, dict]:
        """
        Get a dict of the set parameters.
        If the work environment was loaded from a motion correction batch item it put the bord_px in the dict.
        Doesn't use any arguments

        :return: parameters dict
        :rtype: dict
        """

        # CNMF kwargs
        cnmfe_kwargs = \
            {

            }
