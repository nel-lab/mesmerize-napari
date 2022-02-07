"""Create simple callback that modifies the line visual."""
from skimage import data
from skimage import measure
import numpy as np
import napari
from napari import Viewer
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer

def napari1d_run(image, shapes: dict):
    viewer = napari.Viewer()
    viewer.add_image(image)

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

    viewer1d = napari_plot.ViewerModel1D()
    viewer1d.axis.y_label = "Intensity"
    viewer1d.axis.x_label = ""
    viewer1d.text_overlay.visible = True
    viewer1d.text_overlay.position = "top_right"

    qt_viewer = QtViewer(viewer1d)


    viewer.window.add_dock_widget(qt_viewer, area="bottom", name="Line Widget")
    napari.run()