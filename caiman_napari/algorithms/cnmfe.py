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
import pickle
from caiman.utils.visualization import get_contours as caiman_get_contours
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman_napari.utils import *
from caiman.summary_images import local_correlations_movie_offline
import os
import traceback
from napari.viewer import Viewer
import time
from time import sleep
from ..utils import _organize_coordinates

def main(batch_path, uuid):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    params = item['params']
    print("cnmfe params:", params)

    #adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
     )

    try:
        fname_new = cm.save_memmap(
            [input_movie_path],
            base_name='memmap_',
            order='C',
            dview=dview
        )

        print('making memmap')
        gSig = params['cnmfe_kwargs']['gSig'][0]

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        downsample_ratio = params['downsample_ratio']
        # in fname new load in memmap order C

        cn_filter, pnr = cm.summary_images.correlation_pnr(
            images[::downsample_ratio], swap_dim=False, gSig=gSig
        )

        if not params['do_cnmfe']:
             pnr_output_path = str(pathlib.Path(batch_path).parent.joinpath(f"{uuid}_pnr.pikl").resolve())
             cn_output_path = str(pathlib.Path(batch_path).parent.joinpath(f"{uuid}_cn_filter.pikl").resolve())

             pickle.dump(cn_filter, open(pnr_output_path, 'wb'), protocol=4)
             pickle.dump(pnr, open(cn_output_path, 'wb'), protocol=4)

             output_file_list = \
                 {
                     'cn': cn_output_path,
                     'pnr': pnr_output_path,
                 }

             print(output_file_list)

             d = dict()
             d.update(
                 {
                     "cnmfe_outputs": output_file_list,
                     "cnmfe_memmap": fname_new,
                     "success": True,
                     "traceback": None
                 }
             )
             print('dict for non-cnmfe:', d)

        else:
            cnmfe_params_dict = \
                {
                    "method_init": 'corr_pnr',
                    "n_processes": n_processes,
                    "only_init": True,    # for 1p
                    "center_psf": True,         # for 1p
                    "normalize_init": False     # for 1p
                }
            tot = {**cnmfe_params_dict, **params['cnmfe_kwargs']}
            cnmfe_params_dict = CNMFParams(params_dict=tot)
            cnm = cnmf.CNMF(
                n_processes=n_processes,
                dview=dview,
                params=cnmfe_params_dict
            )
            print("Performing CNMFE")
            cnm = cnm.fit(images)
            print("evaluating components")
            cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

            output_path = str(pathlib.Path(batch_path).parent.joinpath(f"{uuid}.hdf5").resolve())
            cnm.save(output_path)

            d = dict()
            d.update(
                {
                    "cnmfe_outputs": output_path,
                    "cnmfe_memmap": fname_new,
                    "success": True,
                    "traceback": None
                }
            )
            print(d)

    except:
        d = {"success": False, "traceback": traceback.format_exc()}
    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # save dataframe to disc
    df.to_pickle(batch_path)

def load_output(viewer, batch_item: pd.Series):
    print('loading outputs of CNMFE')
    path = batch_item["outputs"].item()["cnmfe_outputs"]
    params = batch_item["params"].item()
    uuid = batch_item['uuid']

    if not params['do_cnmfe']:
        cn_filter = pd.read_pickle(path['cn'])
        viewer.add_image(cn_filter)
        pnr_filter = pd.read_pickle(path['pnr'])
        viewer.add_image(pnr_filter)
    else:
        cnmfe_obj = load_CNMF(path)
        print(cnmfe_obj)
        dims = cnmfe_obj.dims
        if dims is None:
            dims = cnmfe_obj.estimates.dims

        dims = dims[1], dims[0]

        contours_good = caiman_get_contours(
            cnmfe_obj.estimates.A[:, cnmfe_obj.estimates.idx_components],
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
        viewer.add_shapes(
            data=contours_good_coordinates,
            shape_type='polygon',
            edge_width=0.5,
            edge_color=colors_contours_good_edge,
            face_color=colors_contours_good_face,
            opacity=0.7,
        )
        if cnmfe_obj.estimates.idx_components_bad is not None and len(cnmfe_obj.estimates.idx_components_bad) > 0:
            contours_bad = caiman_get_contours(
                cnmfe_obj.estimates.A[:, cnmfe_obj.estimates.idx_components_bad],
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

            viewer.add_shapes(
                data=contours_bad_coordinates,
                shape_type='polygon',
                edge_width=0.5,
                edge_color=colors_contours_bad_edge,
                face_color=colors_contours_bad_face,
                opacity=0.7,
            )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])