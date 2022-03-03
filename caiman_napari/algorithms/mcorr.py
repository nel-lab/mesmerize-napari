import traceback

import click
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import MotionCorrect
from caiman.summary_images import local_correlations_movie_offline
from caiman.utils.utils import download_demo
import psutil
import json
from tqdm import tqdm
import sys
from napari.utils import io
import pandas as pd
import os
from pathlib import Path



@click.command()
@click.option('--batch-path', type=str)
@click.option('--uuid', type=str)
@click.option('--data-path',)
def main(batch_path, uuid, data_path: str = None):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    if data_path is not None:
        data_path = Path(data_path)
        input_movie_path = data_path.joinpath(input_movie_path)

    input_movie_path = str(input_movie_path)

    params = item['params']

    # adapted from current demo notebook
    if 'MESMERIZE_N_PROCESSES' in os.environ.keys():
        try:
            n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
        except:
            n_processes = psutil.cpu_count() - 1
    else:
        n_processes = psutil.cpu_count() - 1

    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    rel_params = dict(params['mcorr_kwargs'])
    opts = CNMFParams(params_dict=rel_params)
    # Run MC, denote boolean 'success' if MC completes w/out error
    try:
        # Run MC
        fnames = [input_movie_path]
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        # Initial code: mc.motion_correct(save_movie=True, base_name_prefix=uuid)
        ## Not sure purpose of the base_name_prefix param
        mc.motion_correct(save_movie=True)
        # Find path to mmap file
        output_path = Path(mc.mmap_file[0])
        if data_path is not None:
            output_path = Path(output_path).relative_to(data_path)

        print("mc finished successfully!")

        print("Computing correlation image")
        Cns = local_correlations_movie_offline([mc.mmap_file[0]],
                                               remove_baseline=True, window=1000, stride=1000,
                                               winSize_baseline=100, quantil_min_baseline=10,
                                               dview=dview)
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        cn_path = str(Path(input_movie_path).parent.joinpath(f'{uuid}_cn.npy'))
        np.save(cn_path, Cn, allow_pickle=False)

        print("finished computing correlation image")

        d = dict()
        d.update(
            {
                "mcorr-output-path": output_path,
                "corr-img-path": Path(cn_path),
                "success": True,
                "traceback": None
            }
        )
    except:
        d = {"success": False, "traceback": traceback.format_exc()}
        print("mc failed, stored traceback in output")

    print(d)
    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # Save DataFrame to disk
    df.to_pickle(batch_path)


def load_projection(viewer, batch_item: pd.Series, proj_type):
    """

    Parameters
    ----------
    viewer: Viewer
        Viewer instance to load the projection into
    batch_item: pd.Series

    proj_type: str
        define type of projection to display {mean, sd, max}

    """
    print("loading projection")
    path = ().joinpath(batch_item['outputs'].item()["mcorr-output-path"])

    Yr, dims, T = cm.load_memmap(str(path))
    MotionCorrectedMovie = np.reshape(Yr.T, [T] + list(dims), order='F')

    MC_Projection = getattr(np, f"nan{proj_type}")(MotionCorrectedMovie, axis=0)

    viewer.add_image(MC_Projection, name=proj_type)


if __name__ == "__main__":
    main()
