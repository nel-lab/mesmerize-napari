from qtpy.QtWidgets import QWidget, QFileDialog, QMessageBox
from qtpy import QtGui
import numpy as np
from matplotlib import cm as matplotlib_color_map
from functools import wraps
import os
from stat import S_IEXEC
import traceback
from typing import *
import re as regex
from pathlib import Path
from warnings import warn

if os.name == "nt":
    IS_WINDOWS = True
    HOME = "USERPROFILE"
else:
    IS_WINDOWS = False
    HOME = "HOME"

def use_open_file_dialog(
    title: str = "Choose file",
    start_dir: Union[str, None] = None,
    exts: List[str] = None,
):
    """
    Use to pass a file path, for opening, into the decorated function using QFileDialog.getOpenFileName
    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    :param exts:        List of file extensions to set the filter in the dialog box
    """

    def wrapper(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            if "qdialog" in kwargs.keys():
                if not kwargs["qdialog"]:
                    func(self, *args, **kwargs)
                    return fn

            if exts is None:
                e = []
            else:
                e = exts

            if isinstance(self, QWidget):
                parent = self
            else:
                parent = None

            path = QFileDialog.getOpenFileName(
                parent, title, os.environ["HOME"], f'({" ".join(e)})'
            )
            if not path[0]:
                return
            path = path[0]
            func(self, path, *args, **kwargs)

        return fn

    return wrapper


def use_save_file_dialog(
    title: str = "Save file", start_dir: Union[str, None] = None, ext: str = None
):
    """
    Use to pass a file path, for saving, into the decorated function using QFileDialog.getSaveFileName
    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    :param exts:        List of file extensions to set the filter in the dialog box
    """

    def wrapper(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            if ext is None:
                raise ValueError("Must specify extension")
            if ext.startswith("*"):
                ex = ext[1:]
            else:
                ex = ext

            if isinstance(self, QWidget):
                parent = self
            else:
                parent = None

            path = QFileDialog.getSaveFileName(parent, title, start_dir, f"(*{ex})")
            if not path[0]:
                return
            path = path[0]
            if not path.endswith(ex):
                path = f"{path}{ex}"

            path = validate_path(path)

            func(self, path, *args, **kwargs)

        return fn

    return wrapper


def use_open_dir_dialog(
    title: str = "Open directory", start_dir: Union[str, None] = None
):
    """
    Use to pass a dir path, to open, into the decorated function using QFileDialog.getExistingDirectory
    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    Example:
    .. code-block:: python
        @use_open_dir_dialog('Select Project Directory', '')
        def load_data(self, path, *args, **kwargs):
            my_func_to_do_stuff_and_load_data(path)
    """

    def wrapper(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            if isinstance(self, QWidget):
                parent = self
            else:
                parent = None

            path = QFileDialog.getExistingDirectory(parent, title)
            if not path:
                return
            func(self, path, *args, **kwargs)

        return fn

    return wrapper


def present_exceptions(
    title: str = "error", msg: str = "The following error occurred."
):
    """
    Use to catch exceptions and present them to the user in a QMessageBox warning dialog.
    The traceback from the exception is also shown.
    This decorator can be stacked on top of other decorators.
    Example:
    .. code-block: python
            @present_exceptions('Error loading file')
            @use_open_file_dialog('Choose file')
                def select_file(self, path: str, *args):
                    pass
    :param title:       Title of the dialog box
    :param msg:         Message to display above the traceback in the dialog box
    :param help_func:   A helper function which is called if the user clicked the "Help" button
    """

    def catcher(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()

                mb = QMessageBox()
                mb.setIcon(QMessageBox.Warning)
                mb.setWindowTitle(title)
                mb.setText(msg)
                mb.setInformativeText(f"{e.__class__.__name__}: {e}")
                mb.setDetailedText(tb)
                mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Help)

        return fn

    return catcher

def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9@\/\\\:._-]*$", str(path)):
        raise ValueError(
            "Paths must only contain alphanumeric characters, "
            "hyphens ( - ), underscores ( _ ) or periods ( . )"
        )
    return path