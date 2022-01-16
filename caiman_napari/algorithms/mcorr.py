import traceback

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
from napari.utils import io
import pandas as pd


def main(batch_path, uuid):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    params = item['params']

    # adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    opts = CNMFParams(params_dict=params)
    # Run MC, denote boolean 'success' if MC completes w/out error
    try:
        # Run MC
        fnames = [str(input_movie_path)]
        mc = MotionCorrect(fnames, dview = dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie = True)
        # Find path to mmap file
        output_path = mc.mmap_file
        d = dict()
        d.update(
            {
                "mcorr_output": output_path,
                "success": True,
                "traceback": None
            }
        )
    except:
        d = {"success": False, "traceback": traceback.format_exc()}

    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # Save DataFrame to disk
    df.to_pickle(batch_path)

def load_output(viewer, batch_item: pd.Series):
    print('Loading outputs of MC')
    path = batch_item['outputs'].item()["mcorr_output"][0]

    Yr, dims, T = cm.load_memmap(path)
    MotionCorrectedMovie = np.reshape(Yr.T, [T] + list(dims), order='F')
    viewer.add_image(MotionCorrectedMovie)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])