"""Create simple callback that modifies the line visual."""
import numpy as np
import napari
from napari.layers import Shapes
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
from mesmerize_core.utils import *
import pandas as pd
from .cnmf_viz_gui import CNMFVizWidget
from .mcorr_viz_gui import MCORRVizWidget
from typing import *


def _get_roi_colormap(self, n_colors) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get colormaps for both face and edges
    """
    edges = auto_colormap(n_colors=n_colors, cmap="hsv", output="mpl")

    faces = auto_colormap(n_colors=n_colors, cmap="hsv", output="mpl", alpha=0.0)

    return edges, faces


class CNMFViewer:
    def __init__(self, batch_item: pd.Series, roi_type: str):
        self.batch_item = batch_item
        self.viewer = napari.Viewer(title="CNMF Visualization")

        self.viz_gui = CNMFVizWidget(cnmf_viewer=self, batch_item=batch_item)
        self.viewer.window.add_dock_widget(
            self.viz_gui, area="bottom", name="Visualization"
        )
        # self.viz_gui.show()
        self.box_size = 10

        # Load correlation map first
        corr_img = batch_item.caiman.get_corr_image()

        self.viewer.add_image(
            corr_img, name=f'corr: {batch_item["item_name"]}', colormap="gray"
        )

        self.cnmf_obj = batch_item.cnmf.get_output()
        self.roi_type = roi_type

        self.napari_spatial_layer_good = None
        self.napari_spatial_layer_bad = None

        self.plot_spatial(component_indices="good")
        self.plot_temporal()

        self.cursor_position = []

    def plot_spatial(self, component_indices: Union[np.ndarray, str] = None):
        if self.roi_type == "outline":
            (
                self.contours_coors,
                self.contours_com,
            ) = self.batch_item.cnmf.get_contours(component_indices=component_indices)

            edge_colors, face_colors = self.get_colors()

            self.spatial_layer: Shapes = self.viewer.add_shapes(
                data=self.contours_coors,
                shape_type="polygon",
                edge_width=0.5,
                edge_color=edge_colors,
                face_color=face_colors,
                opacity=0.7,
                name="good components",
            )

            @self.spatial_layer.mouse_drag_callbacks.append
            def callback(layer, event):
                self.cursor_position = self.viewer.cursor.position
                print(f"global coor position: {self.cursor_position}")
                self.select_contours()

        elif self.roi_type == "mask":
            masks_good = self.batch_item.cnmf.get_spatial_masks(
                self.cnmf_obj.estimates.idx_components
            )
            masks_bad = self.batch_item.cnmf.get_spatial_masks(
                self.cnmf_obj.estimates.idx_components_bad
            )

            edge_colors, face_colors = self.get_colors(alpha_edge=0.0, alpha_face=0.5)

            for i in range(len(masks_good)):
                self.viewer.add_labels(
                    data=masks_good[:, :, i], opacity=0.5, color=edge_colors[i]
                )

            # for i in range(len(masks_bad)):
            #     viewer.add_labels(data=masks_bad[:, :, i], color=masks_bad[i])

            # self.viewer.add_labels(
            #     data=masks_good,
            #     color=face_colors
            # )

            # viewer.add_labels(
            #     data=masks_bad,
            #     # color=colors_bad
            # )

    def update_visible_components(self):
        # Get updated edge and face colormap
        edge_colors, face_colors = self.get_colors()
        # find the original & current number of components
        current_num_components, og_num_components = np.shape(edge_colors)[0], np.shape(self.spatial_layer.edge_color)[0]
        # get indeces of good components
        ixs_comps = self.cnmf_obj.estimates.idx_components
        # if original and updated component numbers don't match, remove existing layer, update input data, replot.
        if og_num_components != current_num_components:
            # Remove and update spatial layer
            self.viewer.layers.remove(self.spatial_layer)
            self.plot_spatial(component_indices=ixs_comps)
            # Remove and update temporal layer
            self.viewer1d.layers.remove(self.temporal_layer)
            self._plot_temporal(ixs_components=ixs_comps)

        else:
            self.spatial_layer.edge_color = edge_colors
            self.temporal_layer.color = edge_colors

    def get_colors(self, alpha_edge=0.8, alpha_face=0.0):
        n_components = len(self.cnmf_obj.estimates.idx_components)

        self.edge_colors = np.vstack(
            auto_colormap(
                n_colors=n_components, cmap="hsv", output="float", alpha=alpha_edge
            )
        )

        self.face_colors = np.vstack(
            auto_colormap(
                n_colors=n_components, cmap="hsv", output="float", alpha=alpha_face
            )
        )
        return self.edge_colors, self.face_colors

    def update_colors(self, sel_comps=None):
        edge_colors, face_colors = self.get_colors()
        edge_colors[:, -1] = 0.0
        if sel_comps is None:
            pass
        else:
            edge_colors[sel_comps, -1] = 0.8

    def show_bad_components(self, b: bool):
        pass

    def _plot_temporal(self, ixs_components: Optional[np.ndarray] = None):
        # extract, format, and plot 'good' traces
        # Traces
        if ixs_components is None:
            ixs = self.cnmf_obj.estimates.idx_components
        else:
            ixs = ixs_components
        good_traces = self.cnmf_obj.estimates.C[ixs]

        edge_colors, face_colors = self.get_colors()

        n_pts = good_traces.shape[1]
        n_lines = good_traces.shape[0]
        xs = [np.linspace(0, n_pts, n_pts)]
        ys = []

        for i in range(n_lines):
            ys.append(good_traces[i])

        self.temporal_layer = self.viewer1d.add_multi_line(
            data=dict(xs=xs, ys=ys), color=edge_colors, name="temporal"
        )

    def plot_temporal(self, ixs_components: Optional[np.ndarray] = None):
        self.viewer1d = napari_plot.Viewer(show=False)
        qt_viewer = QtViewer(self.viewer1d)
        self.viewer1d.axis.y_label = "Intensity"
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.text_overlay.visible = True
        self.viewer1d.text_overlay.position = "top_right"
        self.viewer1d.text_overlay.font_size = 15

        self.viewer.window.add_dock_widget(qt_viewer, area="bottom", name="Line Widget")

        self._plot_temporal(ixs_components=ixs_components)
        # Create layer for infinite line
        self.cnmf_infline_layer = self.viewer1d.add_inf_line(
            data=[1], orientation="vertical", color="red", width=3, name="cnmf slider"
        )
        self.cnmf_infline_layer.move(index=0, pos=[1])
        self.viewer.dims.events.current_step.connect(self.update_cnmf_slider)

    def update_cnmf_slider(self, event):
        time = self.viewer.dims.current_step[0]
        print(time)
        self.cnmf_infline_layer.move(index=0, pos=[time])

    def select_contours(self, box_size=None, update_box=False):
        if update_box:
            try:
                self.viewer.layers.remove(self.white_layer)
            except:
                print("White Layer doesn't exist")
        if box_size is None:
            pass
        else:
            self.box_size = box_size

        sel_comps = [
            ind
            for (ind, x) in enumerate(self.contours_com)
            if (x[1] > self.cursor_position[1] - self.box_size)
            and (x[1] < self.cursor_position[1] + self.box_size)
            and (x[0] > self.cursor_position[0] - self.box_size)
            and (x[0] < self.cursor_position[0] + self.box_size)
        ]

        sel_coors = [self.contours_coors[i] for i in sel_comps]
        sel_coms = [self.contours_com[i] for i in sel_comps]
        print("selected coordinates:", sel_coors)
        face_color = [self.face_colors[i] for i in sel_comps]

        self.update_colors(sel_comps=sel_comps)
        self.temporal_layer.color = self.edge_colors

        box_coors = list()
        box_coors.append([self.cursor_position[0]-self.box_size, self.cursor_position[1]-self.box_size])
        box_coors.append([self.cursor_position[0]+self.box_size, self.cursor_position[1]+self.box_size])


        if len(sel_coors) > 0:
            self.white_layer: Shapes = self.viewer.add_shapes(
                data=sel_coors,
                shape_type="polygon",
                edge_width=0.8,
                edge_color="white",
                face_color=face_color,
                opacity=0.7,
                name="Selected Components",
            )

            self.white_layer.add_rectangles(
                data=box_coors,
                edge_width=0.8,
                edge_color="white",
                face_color=face_color[0]
            )

            @self.white_layer.mouse_drag_callbacks.append
            def callback(layer, event):
                self.cursor_position = self.viewer.cursor.position
                self.viewer.layers.remove(self.white_layer)
                self.select_contours()


class MCORRViewer:
    def __init__(self, batch_item: pd.Series):
        self.batch_item = batch_item
        self.viewer = napari.Viewer(title="MCORR Visualization")
        self.viz_gui = MCORRVizWidget(mcorr_viewer=self, batch_item=batch_item)
        self.viewer.window.add_dock_widget(
            self.viz_gui, area="bottom", name="Visualization"
        )

        # Load input movie optional: Create checkbox
        # Load correlation map first
        corr_img = batch_item.caiman.get_corr_image()

        # self.viewer.add_image(corr_img, name=f'corr: {batch_item["name"]}', colormap='gray')

        self.mcorr_obj = batch_item.mcorr.get_output()
        self.viewer.add_image(
            self.mcorr_obj, name=f'MC Movie: {batch_item["item_name"]}', colormap="gray"
        )

        # plot shifts
        if batch_item["params"]["mcorr_kwargs"]["pw_rigid"] == False:
            self.plot_rig_shifts()
        else:
            self.plot_els_shifts()

    def plot_rig_shifts(self):
        xs, ys = self.batch_item.mcorr.get_shifts(pw_rigid=False)

        self.viewer1d = napari_plot.Viewer(show=False)
        qt_viewer = QtViewer(self.viewer1d)
        self.viewer1d.axis.y_label = "Pixels"
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.text_overlay.visible = True
        self.viewer1d.text_overlay.position = "top_right"
        self.viewer1d.text_overlay.font_size = 15

        n_lines = np.shape(ys)[0]

        self.temporal_layer = self.viewer1d.add_multi_line(
            data=dict(xs=xs, ys=ys),
            color=self.get_colors(n_components=n_lines),
            name="temporal",
        )

        self.viewer.window.add_dock_widget(qt_viewer, area="bottom", name="Line Widget")

        # Create layer for infinite line
        self.mcorr_infline_layer = self.viewer1d.add_inf_line(
            data=[1], orientation="vertical", color="red", width=3, name="mcorr slider"
        )
        self.mcorr_infline_layer.move(index=0, pos=[1])
        self.viewer.dims.events.current_step.connect(self.update_slider)

    def plot_els_shifts(self):
        xs, ys = self.batch_item.mcorr.get_shifts(pw_rigid=True)

        self.viewer1d = napari_plot.Viewer(show=False)
        qt_viewer = QtViewer(self.viewer1d)
        self.viewer1d.axis.y_label = "Pixels"
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.text_overlay.visible = True
        self.viewer1d.text_overlay.position = "top_right"
        self.viewer1d.text_overlay.font_size = 15

        n_lines = np.shape(ys)[0]

        self.temporal_layer = self.viewer1d.add_multi_line(
            data=dict(xs=xs, ys=ys),
            color=self.get_colors(n_components=n_lines),
            name="temporal",
        )

        self.viewer.window.add_dock_widget(qt_viewer, area="bottom", name="Line Widget")

        # Create layer for infinite line
        self.mcorr_infline_layer = self.viewer1d.add_inf_line(
            data=[1], orientation="vertical", color="red", width=3, name="mcorr slider"
        )
        self.mcorr_infline_layer.move(index=0, pos=[1])
        self.viewer.dims.events.current_step.connect(self.update_mcorr_slider)

    def get_colors(self, n_components):
        colors = np.vstack(
            auto_colormap(n_colors=n_components, cmap="hsv", output="float", alpha=1)
        )
        return colors

    def update_mcorr_slider(self, event):
        time = self.viewer.dims.current_step[0]
        print(time)
        self.mcorr_infline_layer.move(index=0, pos=[time])
