"""Create simple callback that modifies the line visual."""
from skimage import data
from skimage import measure
import numpy as np
import napari
from napari import Viewer
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.utils import load_dict_from_hdf5
from caiman_napari.utils import *
import caiman as cm
import pandas as pd
import pyqtgraph as pg

def napari1d_run(batch_item: pd.Series, shapes: dict):
    viewer = napari.Viewer()
    ## Load correlation image
    # Get cnmf memmap
    fname_new = batch_item["outputs"].item()["cnmf_memmap"]
    # Get order f images
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # Get correlation map
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    # Display Correlation Image in viewer
    viewer.add_image(Cn, name="Correlation Image")
    # Display video in viewer
    viewer.add_image(images, name="Movie")
    # Load cnmf file
    path = batch_item["outputs"].item()["cnmf_outputs"]
    cnmf_obj = load_CNMF(path)


    viewer.add_shapes(
        data=shapes['contours_good_coordinates'],
        shape_type='polygon',
        edge_width=0.5,
        edge_color=shapes['colors_contours_good_edge'],
        face_color=shapes['colors_contours_good_face'],
        opacity=0.7,
        name='good components',
    )

    viewer.add_shapes(
        data=shapes['contours_bad_coordinates'],
        shape_type='polygon',
        edge_width=0.5,
        edge_color=shapes['colors_contours_bad_edge'],
        face_color=shapes['colors_contours_bad_face'],
        opacity=0.7,
        name='bad components',
    )

    # Traces
    good_traces = cnmf_obj.estimates.C[cnmf_obj.estimates.idx_components]
    bad_traces = cnmf_obj.estimates.C[cnmf_obj.estimates.idx_components_bad]

    print("good traces", np.shape(good_traces))
    print("bad traces", np.shape(bad_traces))

    viewer1d = napari_plot.ViewerModel1D()
    qt_viewer = QtViewer(viewer1d)
    viewer1d.axis.y_label = "Intensity"
    viewer1d.axis.x_label = ""
    viewer1d.text_overlay.visible = True
    viewer1d.text_overlay.position = "top_right"
    viewer1d.text_overlay.font_size = 15

    # Confirmed the time variable updates real time
    @viewer.dims.events.current_step.connect
    def update_slider(event):
        time = viewer.dims.current_step[0]
        viewer.text_overlay.text = f"{time:1.1f} time"
        print("time", time)
        #viewer1d.add_inf_line

    lines = []
    for i in range(np.shape(good_traces)[0]):
        y = good_traces[i,:]
        lines.append(viewer1d.add_line(np.c_[np.arange(len(y)), y], name=str(i)))
    for i in range(np.shape(bad_traces)[0]):
        y = bad_traces[i,:]
        lines.append(viewer1d.add_line(np.c_[np.arange(len(y)), y], name=str(i)))


    viewer.window.add_dock_widget(qt_viewer, area="bottom", name="Line Widget")

    napari.run()

