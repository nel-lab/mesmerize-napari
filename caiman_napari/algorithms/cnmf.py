"""Performs CNMF in a separate process"""
import pathlib

import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo
import psutil
import json
from tqdm import tqdm
import sys
import pandas as pd

from caiman.utils.visualization import get_contours as caiman_get_contours
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman_napari.utils import *
from caiman.summary_images import local_correlations_movie_offline
import os
import traceback
from napari.viewer import Viewer
import napari_plot

if __name__ != '__main__':
    from ..napari1d_manager import napari1d_run


def main(batch_path, uuid):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    params = item['params']
    print("_cnmf params", params)

    # adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    # merge cnmf and eval kwargs into one dict
    c = dict(params['cnmf_kwargs'])
    e = dict(params['eval_kwargs'])
    tot = {**c, **e}
    cnmf_params = CNMFParams(params_dict=tot)
    # Run CNMF, denote boolean 'success' if CNMF completes w/out error
    try:
        fname_new = cm.save_memmap(
            [input_movie_path],
            base_name='memmap_',
            order='C',
            dview=dview
        )

        print('making memmap')

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        # in fname new load in memmap order C

        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local',
            n_processes=None,
            single_thread=False
        )

        print("performing CNMF")
        cnm = cnmf.CNMF(
            n_processes,
            params=cnmf_params,
            dview=dview
        )

        print("fitting images")
        cnm = cnm.fit(images)
        #
        if params['refit'] == True:
            print('refitting')
            cnm = cnm.refit(images, dview=dview)

        print("Eval")
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

        output_path = str(pathlib.Path(batch_path).parent.joinpath(f"{uuid}.hdf5").resolve())

        cnm.save(output_path)
        d = dict()
        d.update(
            {
                "cnmf_outputs": output_path,
                "cnmf_memmap": fname_new,
                "success": True,
                "traceback": None
            }
        )
    except:
        d = {"success": False, "traceback": traceback.format_exc()}
    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # save dataframe to disc
    df.to_pickle(batch_path)


def load_output(viewer, batch_item: pd.Series):
    print('Loading outputs of CNMF')
    path = batch_item["outputs"].item()["cnmf_outputs"]
    cnmf_obj = load_CNMF(path)

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

    colors_contours_good_edge = auto_colormap(
        n_colors=len(contours_good),
        cmap='hsv',
        output='mpl',
    )
    colors_contours_good_face = auto_colormap(
        n_colors=len(contours_good),
        cmap='hsv',
        output='mpl',
        alpha=0.0,
    )

    contours_good_coordinates = [_organize_coordinates(c) for c in contours_good]

    if cnmf_obj.estimates.idx_components_bad is not None and len(cnmf_obj.estimates.idx_components_bad) > 0:
        contours_bad = caiman_get_contours(
            cnmf_obj.estimates.A[:, cnmf_obj.estimates.idx_components_bad],
            dims,
            swap_dim=True
        )

        contours_bad_coordinates = [_organize_coordinates(c) for c in contours_bad]

        colors_contours_bad_edge = auto_colormap(
            n_colors=len(contours_bad),
            cmap='hsv',
            output='mpl',
        )
        colors_contours_bad_face = auto_colormap(
            n_colors=len(contours_bad),
            cmap='hsv',
            output='mpl',
            alpha=0.0
        )
    # create dictionary for shapes of contours to pass to napari1d manager
    shapes_dict = \
        {
            'contours_bad_coordinates': contours_bad_coordinates,
            'colors_contours_bad_edge': colors_contours_bad_edge,
            'colors_contours_bad_face': colors_contours_bad_face,
            'contours_good_coordinates': contours_good_coordinates,
            'colors_contours_good_edge': colors_contours_good_edge,
            'colors_contours_good_face': colors_contours_good_face,
        }

    napari1d_run(batch_item=batch_item, shapes=shapes_dict)


def _organize_coordinates(contour: dict):
    coors = contour['coordinates']
    coors = coors[~np.isnan(coors).any(axis=1)]

    return coors


def load_projection(viewer, batch_item: pd.Series, proj_type):
    """
    Load correlation map from cnmf memmap file

    Parameters
    ----------
    viewer: Viewer
        Viewer instance to load the projection in

    batch_item: pd.Series

    proj_type: None
        Not used

    """
    # Get cnmf memmap
    fname_new = batch_item["outputs"].item()["cnmf_memmap"]
    # Get order f images
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # Get correlation map
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    # Add correlation map to napari viewer
    viewer.add_image(Cn)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
