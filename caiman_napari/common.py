from qtpy.QtWidgets import QWidget, QFileDialog
from qtpy import QtGui
import numpy as np
from matplotlib import cm as matplotlib_color_map
from functools import wraps
import os
from typing import *

# Useful functions from mesmerize


qualitative_colormaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
              'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']


def use_open_file_dialog(title: str = 'Choose file', start_dir: Union[str, None] = None, exts: List[str] = None):
    """
    Use to pass a file path, for opening, into the decorated function using QFileDialog.getOpenFileName

    :param title:       Title of the dialog box
    :param start_dir:   Directory that is first shown in the dialog box.
    :param exts:        List of file extensions to set the filter in the dialog box
    """
    def wrapper(func):

        @wraps(func)
        def fn(self, *args, **kwargs):
            if 'qdialog' in kwargs.keys():
                if not kwargs['qdialog']:
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

            path = QFileDialog.getOpenFileName(parent, title, os.environ['HOME'], f'({" ".join(e)})')
            if not path[0]:
                return
            path = path[0]
            func(self, path, *args, **kwargs)
        return fn
    return wrapper


def auto_colormap(
        n_colors: int,
        cmap: str = 'hsv',
        output: str = 'mpl',
        spacing: str = 'uniform',
        alpha: float = 1.0
    ) \
        -> List[Union[QtGui.QColor, np.ndarray, str]]:
    """
    If non-qualitative map: returns list of colors evenly spread through the chosen colormap.
    If qualitative map: returns subsequent colors from the chosen colormap

    :param n_colors: Numbers of colors to return
    :param cmap:     name of colormap

    :param output:   option: 'mpl' returns RGBA values between 0-1 which matplotlib likes,
                     option: 'bokeh' returns hex strings that correspond to the RGBA values which bokeh likes

    :param spacing:  option: 'uniform' returns evenly spaced colors across the entire cmap range
                     option: 'subsequent' returns subsequent colors from the cmap

    :param alpha:    alpha level, 0.0 - 1.0

    :return:         List of colors as either ``QColor``, ``numpy.ndarray``, or hex ``str`` with length ``n_colors``
    """

    valid = ['mpl', 'pyqt', 'bokeh']
    if output not in valid:
        raise ValueError(f'output must be one {valid}')

    valid = ['uniform', 'subsequent']
    if spacing not in valid:
        raise ValueError(f'spacing must be one of either {valid}')

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be within 0.0 and 1.0')

    cm = matplotlib_color_map.get_cmap(cmap)
    cm._init()

    if output == 'pyqt':
        lut = (cm._lut * 255).view(np.ndarray)
    else:
        lut = (cm._lut).view(np.ndarray)

    lut[:, 3] *= alpha

    if spacing == 'uniform':
        if not cmap in qualitative_colormaps:
            cm_ixs = np.linspace(0, 210, n_colors, dtype=int)
        else:
            if n_colors > len(lut):
                raise ValueError('Too many colors requested for the chosen cmap')
            cm_ixs = np.arange(0, len(lut), dtype=int)
    else:
        cm_ixs = range(n_colors)

    colors = []
    for ix in range(n_colors):
        c = lut[cm_ixs[ix]]

        if output == 'bokeh':
            c = tuple(c[:3] * 255)
            hc = '#%02x%02x%02x' % tuple(map(int, c))
            colors.append(hc)

        else:  # mpl
            colors.append(c)

    return colors