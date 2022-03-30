import pathlib
import click
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import pandas as pd
import pickle
import traceback
from napari.viewer import Viewer
from pathlib import Path

if __name__ == '__main__':
    from mesmerize_napari.core import set_parent_data_path, get_full_data_path

@click.command()
@click.option('--batch-path', type=str)
@click.option('--uuid', type=str)
@click.option('--data-path')
def main(batch_path, uuid, data_path: str = None):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    set_parent_data_path(data_path)
    input_movie_path = str(get_full_data_path(input_movie_path))

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
             pnr_output_path = str(Path(batch_path).parent.joinpath(f"{uuid}_pnr.pikl").resolve())
             cn_output_path = str(Path(batch_path).parent.joinpath(f"{uuid}_cn_filter.pikl").resolve())

             pickle.dump(cn_filter, open(pnr_output_path, 'wb'), protocol=4)
             pickle.dump(pnr, open(cn_output_path, 'wb'), protocol=4)

             if data_path is not None:
                 pnr_output_path = Path(pnr_output_path).relative_to(data_path)
                 cn_output_path = Path(cn_output_path).relative_to(data_path)
                 cnmfe_memmap_path = Path(fname_new).relative_to(data_path)
             else:
                 cnmfe_memmap_path = fname_new

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
                     "cnmfe_memmap": cnmfe_memmap_path,
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

            output_path = str(Path(batch_path).parent.joinpath(f"{uuid}.hdf5").resolve())
            cnm.save(output_path)

            if data_path is not None:
                output_path = Path(output_path).relative_to(data_path)
                cnmfe_memmap_path = Path(fname_new).relative_to(data_path)
            else:
                output_path = Path(output_path)
                cnmfe_memmap_path = Path(fname_new)

            d = dict()
            d.update(
                {
                    "cnmfe_outputs": output_path,
                    "cnmfe_memmap": cnmfe_memmap_path,
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
    fname_new = batch_item["outputs"].item()["cnmfe_memmap"]
    # Get order f images
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # Get correlation map
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    # Add correlation map to napari viewer
    viewer.add_image(Cn, name="Correlation Map (1P)")


if __name__ == "__main__":
    main()