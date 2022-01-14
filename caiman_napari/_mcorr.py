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

    # Run MC
    fnames = [str(input_movie_path)]
    mc = MotionCorrect(fnames, dview = dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie = True)
    ix = df[df['uuid'] == uuid].index[0]
    df['outputs'][ix] = mc.mmap_file
    # Add the Series to the DataFrame
    print('map file', df['outputs'][ix])
    print('new df', df)
    # Save DataFrame to disk
    df.to_pickle(batch_path)
    #np.save(input_movie_path + 'mc.npy', mc.mmap_file)

def load_output_mcorr(viewer, batch_item: pd.Series):
    print("output mcorr movie")
    path = batch_item['outputs'][0]
    print(path)
    Yr, dims, T = cm.load_memmap(path)
    MotionCorrectedMovie = np.reshape(Yr.T, [T] + list(dims), order='F')
    viewer.add_image(MotionCorrectedMovie)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])