"""Create simple callback that modifies the line visual."""
from skimage import data
from skimage import measure
import numpy as np
import napari
from napari import Viewer
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
from qtpy.QtWidgets import QVBoxLayout
from caiman import load_memmap
from caiman_napari.utils import *
import caiman as cm
import pandas as pd
import pyqtgraph as pg
from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
from .core import CaimanSeriesExtensions, CNMFExtensions
from tqdm import tqdm


def _get_roi_colormap(self, n_colors) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get colormaps for both face and edges
    """
    edges = auto_colormap(
        n_colors=n_colors,
        cmap='hsv',
        output='mpl'
    )

    faces = auto_colormap(
        n_colors=n_colors,
        cmap='hsv',
        output='mpl',
        alpha=0.0
    )

    return edges, faces


def napari1d_run(batch_item: pd.Series, roi_type: str):
    viewer = napari.Viewer(title="CNMF Visualization")
    ## Load correlation image
    # Get cnmf memmap
    # fname_new = batch_item["outputs"].item()["cnmf-memmap"]
    # # Get order f images
    # Yr, dims, T = cm.load_memmap(fname_new)
    # images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # # Get correlation map
    # Cn = cm.local_correlations(images.transpose(1, 2, 0))
    # Cn[np.isnan(Cn)] = 0
    # Display Correlation Image in viewer
    # viewer.add_image(Cn, name="Correlation Image")
    # Display video in viewer
    movie_path = str(batch_item.caiman.get_input_movie_path())
    if movie_path.endswith('mmap'):
        Yr, dims, T = load_memmap(movie_path)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        viewer.add_image(images, name="Movie")
    else:
        viewer.open(movie_path)
    # Load cnmf file
    # path = batch_item["outputs"].item()["cnmf_hdf5"]
    # cnmf_obj = load_CNMF(path)

    cnmf_obj = batch_item.cnmf.get_output()

    colors_good = auto_colormap(
        n_colors=len(cnmf_obj.estimates.idx_components),
        cmap='hsv',
        output='mpl'
    )

    colors_good_zero_alpha = auto_colormap(
        n_colors=len(cnmf_obj.estimates.idx_components),
        cmap='hsv',
        output='mpl',
        alpha=0.0
    )

    colors_bad = auto_colormap(
        n_colors=len(cnmf_obj.estimates.idx_components_bad),
        cmap='hsv',
        output='mpl'
    )

    colors_bad_zero_alpha = auto_colormap(
        n_colors=len(cnmf_obj.estimates.idx_components_bad),
        cmap='hsv',
        output='mpl',
        alpha=0.0
    )

    if roi_type == 'outline':
        coors_good = batch_item.cnmf.get_spatial_contour_coors(cnmf_obj.estimates.idx_components)
        coors_bad = batch_item.cnmf.get_spatial_contour_coors(cnmf_obj.estimates.idx_components_bad)

        viewer.add_shapes(
            data=coors_good,
            shape_type='polygon',
            edge_width=0.5,
            edge_color=colors_good,
            face_color=colors_good_zero_alpha,
            opacity=0.7,
            name='good components',
        )

        viewer.add_shapes(
            data=coors_bad,
            shape_type='polygon',
            edge_width=0.5,
            edge_color=colors_bad,
            face_color=colors_bad_zero_alpha,
            opacity=0.7,
            name='bad components',
        )

    elif roi_type == 'mask':
        masks_good = batch_item.cnmf.get_spatial_masks(cnmf_obj.estimates.idx_components)
        masks_bad = batch_item.cnmf.get_spatial_masks(cnmf_obj.estimates.idx_components_bad)

        for i in range(len(masks_good)):
            viewer.add_labels(data=masks_good[:, :, i])#, color=colors_good[i])

        # for i in range(len(masks_bad)):
        #     viewer.add_labels(data=masks_bad[:, :, i], color=masks_bad[i])

        # viewer.add_labels(
        #     data=masks_good,
        #     # color=colors_good
        # )
        #
        # viewer.add_labels(
        #     data=masks_bad,
        #     # color=colors_bad
        # )

    # Traces
    good_traces = cnmf_obj.estimates.C[cnmf_obj.estimates.idx_components]
    bad_traces = cnmf_obj.estimates.C[cnmf_obj.estimates.idx_components_bad]

    print("good traces", np.shape(good_traces))
    print("bad traces", np.shape(bad_traces))

    viewer1d = napari_plot.ViewerModel1D()
    qt_viewer = QtViewer(viewer1d)
    viewer1d.axis.y_label = "Intensity"
    viewer1d.axis.x_label = "Time"
    viewer1d.text_overlay.visible = True
    viewer1d.text_overlay.position = "top_right"
    viewer1d.text_overlay.font_size = 15
    # Create layer for infinite line
    infline_layer = viewer1d.add_inf_line(data=[1], orientation="vertical", color="red", width=3, name="slider")
    infline_layer.move(index=0, pos=[1000])
    viewer1d.add_layer(layer=infline_layer)

    viewer1d.layers.toggle_selected_visibility()

    lines = []
    for i in tqdm(range(np.shape(good_traces)[0])):
        y = good_traces[i,:]
        lines.append(viewer1d.add_line(np.c_[np.arange(len(y)), y], name=str(i), color=colors_good[i]))
    for i in tqdm(range(np.shape(bad_traces)[0])):
        y = bad_traces[i,:]
        lines.append(viewer1d.add_line(np.c_[np.arange(len(y)), y], name=str(i), color=colors_bad[i]))
    viewer.window.add_dock_widget(qt_viewer, area="bottom", name="Line Widget")

    @viewer1d.bind_key('n')
    def print_names(viewer1d):
        print([layer.name for layer in viewer1d.layers])
        viewer1d.layers.enabled = True
        qt_viewer.on_toggle_controls_dialog()

    @viewer.dims.events.current_step.connect
    def update_slider(event):
        time = viewer.dims.current_step[0]
        print(time)
        infline_layer.move(index=0, pos=[time])
