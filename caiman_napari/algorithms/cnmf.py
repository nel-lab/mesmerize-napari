"""Performs CNMF in a separate process"""
import pathlib
import click
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import pandas as pd
from caiman_napari.utils import *
from caiman.summary_images import local_correlations_movie_offline
import traceback
from napari.viewer import Viewer



@click.command()
@click.option('--batch-path', type=str)
@click.option('--uuid', type=str)
@click.option('--data-path')
def main(batch_path, uuid, data_path: str = None):
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
                "cnmf-hdf5-path": output_path,
                "cnmf-memmap-path": fname_new,
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
    viewer.add_image(Cn, name="Correlation Map")


if __name__ == "__main__":
    main()
