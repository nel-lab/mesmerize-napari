import time
from PyQt5.uic.properties import QtGui
from .main_offline_gui_template import Ui_MainOfflineGUIWidget
from .mcorr_gui import MCORRWidget
from .cnmf_gui import CNMFWidget
from .cnmfe_gui import CNMFEWidget
from PyQt5 import QtWidgets
from qtpy import QtWidgets
from napari_plugin_engine import napari_hook_implementation
from napari import Viewer
from mesmerize_core.utils import *
from mesmerize_core import *
from .napari_extensions import CaimanNapariSeriesExtensions
import pandas as pd
from functools import partial
import pprint
import caiman as cm
import numpy as np
import psutil
from .napari1d_manager import CNMFViewer, MCORRViewer
from pathlib import Path
from PyQt5 import QtCore
import matplotlib.pyplot as plt
from .pyqt_decorators import *

if not IS_WINDOWS:
    from signal import SIGKILL

elif IS_WINDOWS:
    from win32api import TerminateProcess, CloseHandle


COLORS_HEX = {
    "orange": "#ffb347",
    "green": "#77dd77",
    "dark-green": "#009603",
    "red": "#fe0d00",
    "blue": "#85e3ff",
    "yellow": "#ffff00",
}


class MainOfflineGUI(QtWidgets.QWidget):
    def __init__(self, napari_viewer: Viewer):
        QtWidgets.QWidget.__init__(self)

        self.viewer = napari_viewer
        self.ui = Ui_MainOfflineGUIWidget()
        self.ui.setupUi(self)
        self.show()

        self.input_movie_path: str = None

        self.dataframe: pd.DataFrame = None
        self.dataframe_file_path: str = None
        # define actions for each button
        # Open Movie
        self.ui.pushButtonOpenMovie.clicked.connect(self.open_movie)
        # Open Panel to set parameters for CNMF
        self.ui.pushButtonParamsCNMF.clicked.connect(self.show_cnmf_params_gui)
        # Open panel to set parameters MCORR
        self.ui.pushButtonParamsMCorr.clicked.connect(self.show_mcorr_params_gui)
        # Open panel to set parameters for CNMFE
        self.ui.pushButtonParamsCNMFE.clicked.connect(self.show_cnmfe_params_gui)
        # Start Batch
        self.ui.pushButtonNewBatch.clicked.connect(self.create_new_batch)
        # Open Batch
        self.ui.pushButtonOpenBatch.clicked.connect(self.open_batch)
        # Start running from zereoth index
        self.ui.pushButtonStart.clicked.connect(self.run)
        # Start running from selected index
        self.ui.pushButtonStartItem.clicked.connect(self.run_item)
        self.ui.pushButtonAbort.clicked.connect(self.abort_run)
        # Remove selected item
        self.ui.pushButtonDelItem.clicked.connect(self.remove_item)
        # Change parameters displayed in param text box upon change in selection
        self.ui.listWidgetItems.currentRowChanged.connect(self.set_params_text)
        # On double click of item in listwidget, load results and display
        self.ui.listWidgetItems.doubleClicked.connect(self.load_output)
        # Show MCorr Projections
        self.ui.pushButtonViewProjection.clicked.connect(self.view_projections)

        self.ui.pushButtonViewInput.clicked.connect(self.view_input)

        self.ui.lineEditParentDataPath.textChanged.connect(self.set_parent_data_path)

        self.qprocess: QtCore.QProcess = None

        self.ui.pushButtonVizCorrelationImage.clicked.connect(
            self.load_correlation_image
        )

        self.ui.pushButtonViewDownsampledMCorrrMovie.clicked.connect(
            self.view_downsample_mcorr
        )

        self.ui.pushButtonViewMCShifts.clicked.connect(self.view_shifts)

        self.ui.pushButtonVizReconstructedMovie.clicked.connect(self.view_reconstructed_movie)

        self.mcorr_params_gui = None
        self.cnmf_params_gui = None
        self.cnmfe_params_gui = None

        # On event of Recent input movie selection, update self.input_movie_path
        self.ui.comboBoxRecentInputMovies.activated.connect(self.set_input_movie_path)
        # self.evaluate_components_window = EvalComponentsWidgets(parent=self)
        # self.ui.pushButtonEvaluateCNMFComponents.clicked.connect(self.evaluate_components_window.show)

    def set_parent_data_path(self):
        path = Path(self.ui.lineEditParentDataPath.text())
        if not path.is_dir():
            self.ui.lineEditParentDataPath.setStyleSheet(
                f"QLineEdit {{background: {COLORS_HEX['red']}}}"
            )
        else:
            self.ui.lineEditParentDataPath.setStyleSheet(
                f"QLineEdit {{background: {COLORS_HEX['dark-green']}}}"
            )
            set_parent_raw_data_path(path)

    @use_open_dir_dialog("Select Parent Data Directory")
    def set_parent_data_path_dialog(self, path):
        self.ui.lineEditParentDataPath.setText(path)
        self.set_parent_raw_data_path()
    def open_movie(self, path: str, *args, **kwargs):
        if not self.clear_viewer():
            return

        self._open_movie(path)
    @use_open_file_dialog("choose file", "", ["*.tiff", "*.tif", "*.btf", "*.mmap"])
    def _open_movie(self, path: str, name: str = None):
        # Add movie to recent movie list, set to self.input_movie_path
        self.ui.comboBoxRecentInputMovies.addItem(f"User Accessed: {path}", str(path))
        self.ui.comboBoxRecentInputMovies.setCurrentIndex(self.ui.comboBoxRecentInputMovies.count()-1)
        self.set_input_movie_path()

        file_ext = Path(self.input_movie_path).suffix
        if file_ext == ".mmap":
            Yr, dims, T = cm.load_memmap(self.input_movie_path)
            images = np.reshape(Yr.T, [T] + list(dims), order="F")
            self.viewer.add_image(images, name=name, colormap="gnuplot2")
        else:
            self.viewer.open(self.input_movie_path, colormap="gnuplot2")

    def set_input_movie_path(self):
        self.input_movie_path = self.ui.comboBoxRecentInputMovies.currentData()

    def view_input(self):
        path = self.selected_series().caiman.get_input_movie_path()
        self._open_movie(path)

    def clear_viewer(self) -> bool:
        if len(self.viewer.layers) == 0:
            return True

        if (
            QtWidgets.QMessageBox.warning(
                self,
                "Clear viewer?",
                "The viewer must be cleared, do you want to continue?",
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No,
            )
            == QtWidgets.QMessageBox.No
        ):
            return False

        for layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

        return True
    @use_save_file_dialog("Choose location to save batch file", "", ".pickle")
    def create_new_batch(self, path: str, *args, **kwargs):
        self.ui.listWidgetItems.clear()
        self.dataframe = create_batch(path)
        self.dataframe_file_path = path

    @use_open_file_dialog("Choose batch", "", ["*.pickle"])
    def open_batch(self, path: str, *args, **kwargs):
        self.dataframe = load_batch(path)
        self.dataframe_file_path = path

        self.ui.listWidgetItems.clear()

        # Iterate through dataframe, add each item to list widget
        ## For now, instead of name I'm adding the uuid
        for i, r in self.dataframe.iterrows():
            algo = r["algo"]
            name = r["item_name"]
            uuid = r["uuid"]
            self.ui.listWidgetItems.addItem(f"{algo}: {name}")

            item = self.ui.listWidgetItems.item(i)
            item.setData(3, uuid)

            self.set_list_widget_item_color(i)

            if algo == 'mcorr' and r["outputs"]:
                if r["outputs"]["success"]:
                    input_movie_path = r.mcorr.get_output_path()
                    self.ui.comboBoxRecentInputMovies.addItem(f"MC Outputs: {input_movie_path}", str(input_movie_path))
        self.ui.comboBoxRecentInputMovies.setCurrentIndex(-1)

    def add_item(
        self, algo: str, parameters: dict, name: str, input_movie_path: str = None
    ):
        if self.dataframe is None:
            QtWidgets.QMessageBox.warning(
                "No Batch", "You must open or create a batch before adding items."
            )
            return

        if self.input_movie_path is None:
            QtWidgets.QMessageBox.warning(
                "No movie open", "You must open a movie to add to the batch."
            )
            return

        # Input movie path should always be shown in input movie box
        input_movie_path = self.input_movie_path

        self.dataframe.caiman.add_item(
            algo=algo, item_name=name, input_movie_path=input_movie_path, params=parameters
        )
        print(f"Added <{algo}> item to batch!")

        uuid = self.dataframe.iloc[-1]["uuid"]

        self.ui.listWidgetItems.addItem(f"{algo}: {name}")

        n = self.ui.listWidgetItems.count()
        item = self.ui.listWidgetItems.item(n - 1)
        item.setData(3, uuid)

    def remove_item(self):
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)

        ix = self.dataframe[self.dataframe["uuid"] == uuid].index[0]
        self.dataframe.caiman.remove_item(index=ix)
        self.ui.listWidgetItems.takeItem(ix)

    def run_item(self):
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)

        ix = self.dataframe[self.dataframe["uuid"] == uuid].index[0]

        # For now, just run given index.
        self._run_index(ix)

    def run(self):
        self._run_index(0)

    def _run_index(self, index: int):
        callbacks = [partial(self.item_finished, index)]
        std_out = self._print_qprocess_std_out

        self.qprocess = self.dataframe.iloc[index].caiman_napari.run(
            callbacks_finished=callbacks,
            callback_std_out=std_out,
        )
        self.set_list_widget_item_color(index, "yellow")

    def _print_qprocess_std_out(self, proc):
        txt = proc.readAllStandardOutput().data().decode("utf8")
        self.ui.textBrowserStdOut.append(txt)

    def item_finished(self, ix):
        self.qprocess = None

        self.dataframe = load_batch(self.dataframe_file_path)
        self.set_list_widget_item_color(ix)

        if (ix + 1) < self.ui.listWidgetItems.count():
            time.sleep(10)
            self._run_index(ix + 1)

        else:
            QtWidgets.QMessageBox.information(
                self, "Batch is done!", "Yay, your batch has finished processing!"
            )

    def abort_run(self):
        if self.qprocess is not None:
            self.qprocess.disconnect()

        # self.qprocess.terminate()

        try:
            py_proc = psutil.Process(self.qprocess.pid()).children()[0].pid
        except psutil.NoSuchProcess:
            return
        children = psutil.Process(py_proc).children()

        if not IS_WINDOWS:
            os.kill(py_proc, SIGKILL)

            for child in children:
                os.kill(child.pid, SIGKILL)

        if IS_WINDOWS:
            TerminateProcess(py_proc, -1)
            CloseHandle(py_proc)

            for child in children:
                TerminateProcess(child.pid, -1)
                CloseHandle(child.pid)

        self.qprocess = None

    def set_params_text(self, ix):
        u = self._selected_uuid()
        s = self.selected_series()

        params = s["params"]
        params_str = pprint.pformat(params)
        input_movie_path = s["input_movie_path"]

        output_str = (
            f"uuid:\n{u}\n"
            f"input_movie_path:\n{input_movie_path}\n\n"
            f"params:\n{params_str}"
        )

        if s["outputs"] is not None:
            if s["outputs"]["traceback"] is not None:
                tb = s["outputs"]["traceback"]
                output_str += f"\n\ntraceback:{tb}"

        self.ui.textBrowserParams.setText(output_str)

    def set_list_widget_item_color(self, ix: int, color: str = None):
        if color is not None:
            self._set_list_widget_item_color(ix, color)

        elif color is None:
            if self.dataframe.iloc[ix]["outputs"] is None:
                return

            if self.dataframe.iloc[ix]["outputs"]["success"]:
                self._set_list_widget_item_color(ix, "green")
                if self.dataframe.iloc[ix]["algo"] == "mcorr":
                    input_movie_path = self.dataframe.iloc[ix].mcorr.get_output_path()
                    self.ui.comboBoxRecentInputMovies.addItem(f"MC Outputs: {input_movie_path}", str(input_movie_path))
            else:
                self._set_list_widget_item_color(ix, "red")

    def _set_list_widget_item_color(self, ix: int, color: str):
        self.ui.listWidgetItems.item(ix).setBackground(
            QtGui.QBrush(QtGui.QColor(COLORS_HEX[color]))
        )

    def show_cnmf_params_gui(self):
        if self.cnmf_params_gui is None:
            self.cnmf_params_gui = CNMFWidget(parent=self)

        self.cnmf_params_gui.show()

    def show_mcorr_params_gui(self):
        if self.mcorr_params_gui is None:
            self.mcorr_params_gui = MCORRWidget(parent=self)

        self.mcorr_params_gui.show()

    def show_cnmfe_params_gui(self):
        if self.cnmfe_params_gui is None:
            self.cnmfe_params_gui = CNMFEWidget(parent=self)

        self.cnmfe_params_gui.show()

    def _selected_uuid(self) -> str:
        # Find uuid for selected item
        item_gui = QtWidgets.QListWidgetItem = self.ui.listWidgetItems.currentItem()
        uuid = item_gui.data(3)
        return uuid

    def selected_series(self) -> pd.Series:
        u = self._selected_uuid()
        return self.dataframe.caiman.uloc(u)

    def load_output(self):
        # clear napari viewer before loading new movies
        if not self.clear_viewer():
            return

        s = self.selected_series()  # pandas series corresponding to the item
        algo = s["algo"]
        if algo == "mcorr":
            MCORRViewer(self.selected_series())

        elif algo in ["cnmf", "cnmfe"]:
            if self.ui.radioButtonROIMask.isChecked():
                CNMFViewer(self.selected_series(), "mask")

            elif self.ui.radioButtonROIOutline.isChecked():
                CNMFViewer(self.selected_series(), "outline")

    def load_correlation_image(self):
        s = self.selected_series()
        corr_img = s.caiman.get_corr_image()
        if s["algo"] == "cnmfe":
            pnr_img = s.caiman.get_pnr_image()
            self.viewer.add_image(
                pnr_img, name=f'pnr: {s["item_name"]}', colormap="gnuplot2"
            )
        self.viewer.add_image(corr_img, name=f'corr: {s["item_name"]}', colormap="gnuplot2")

    def view_projections(self):
        proj_type = self.ui.comboBoxProjectionOpts.currentText()
        s = self.selected_series()
        projection = s.caiman.get_projection(proj_type=proj_type)
        self.viewer.add_image(
            projection, name=f'{proj_type}: projection {s["item_name"]}', colormap="gnuplot2"
        )

    def view_downsample_mcorr(self):
        s = self.selected_series()
        downsample_window = self.ui.spinBoxDownsampleWindow.value()
        self.ds_video = s.mcorr.get_output()
        self.viewer.add_image(
            self.ds_video,
            name='MC Movie'
        )
        frame0 = np.nanmean(self.ds_video[0:downsample_window], axis=0)
        self.viewer.add_image(
                    frame0,
                    name='Downsampled MC Movie')
        self.viewer.dims.events.current_step.connect(self.update_ds_slider)

    def update_ds_slider(self, event):
        downsample_window = self.ui.spinBoxDownsampleWindow.value()
        ix = self.viewer.dims.current_step[0]
        start = max(0, ix - downsample_window)
        end = min(self.ds_video.shape[0], ix + downsample_window)
        ds_frame = np.nanmean(self.ds_video[start:end], axis=0)
        self.viewer.layers['Downsampled MC Movie'].data = ds_frame

    def view_shifts(self):
        s = self.selected_series()
        if s["params"]['mcorr_kwargs']["pw_rigid"]:
            xs, ys = s.mcorr.get_shifts(pw_rigid=True)

            plt.figure()
            n_lines = int(np.shape(ys)[0]/2)
            print(n_lines)
            for i in range(int(n_lines)):
                plt.plot(xs[0], ys[i])
            plt.title("Elastic MC Shifts (x)")
            plt.xlabel("Time")
            plt.ylabel("Pixels")

            plt.figure()
            for i in range(n_lines):
                plt.plot(xs[0], ys[n_lines+i])
            plt.title("Elastic MC Shifts (y)")
            plt.xlabel("Time")
            plt.ylabel("Pixels")

        else:
            xs, ys = s.mcorr.get_shifts(pw_rigid=False)

            plt.figure()
            for i in range(np.shape(ys)[0]):
                plt.plot(xs[0], ys[i])

            plt.title("Rigid MC Shifts")
            plt.legend(["x-shifts", "y-shifts"])
            plt.xlabel("Time")
            plt.ylabel("Pixels")
    def view_reconstructed_movie(self):
        s = self.selected_series()
        rcm = s.cnmf.get_rcm(
            component_indices="good",
        )
        self.viewer.add_image(
            rcm,
            name='Reconstructed Movie'
        )


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MainOfflineGUI
