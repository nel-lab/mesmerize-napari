from caiman.utils.visualization import get_contours as caiman_get_contours
from mesmerize_napari.core.utils import *


def show_results(cnmf_obj, viewer):
    dims = cnmf_obj.dims
    if dims is None:  # I think that one of these is `None` if loaded from an hdf5 file
        dims = cnmf_obj.estimates.dims

    # need to transpose these
    dims = dims[1], dims[0]

    contours_good = caiman_get_contours(
        cnmf_obj.estimates.A[:, cnmf_obj.estimates.idx_components],
        dims,
        swap_dim=True
    )

    colors_contours_good = auto_colormap(
        n_colors=len(contours_good),
        cmap='hsv',
        output='mpl',
    )

    contours_good_coordinates = [_organize_coordinates(c) for c in contours_good]
    viewer.add_shapes(
        data=contours_good_coordinates,
        shape_type='polygon',
        edge_width=0.5,
        edge_color=colors_contours_good,
        face_color=colors_contours_good,
        opacity=0.1,
    )

    if cnmf_obj.estimates.idx_components_bad is not None and len(cnmf_obj.estimates.idx_components_bad) > 0:
        contours_bad = caiman_get_contours(
            cnmf_obj.estimates.A[:, cnmf_obj.estimates.idx_components_bad],
            dims,
            swap_dim=True
        )

        contours_bad_coordinates = [_organize_coordinates(c) for c in contours_bad]

        colors_contours_bad = auto_colormap(
            n_colors=len(contours_bad),
            cmap='hsv',
            output='mpl',
        )

        viewer.add_shapes(
            data=contours_bad_coordinates,
            shape_type='polygon',
            edge_width=0.5,
            edge_color=colors_contours_bad,
            face_color=colors_contours_bad,
            opacity=0.1,
        )

def _organize_coordinates(contour: dict):
    coors = contour['coordinates']
    coors = coors[~np.isnan(coors).any(axis=1)]

    return coors