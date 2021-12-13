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

# Parameters for Motion Correction
# dataset dependent parameters
fr = 30  # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds
dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
# note the lower than usual spatial resolution here
max_shift_um = (25., 25.)  # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = False  # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
# overlap between pathes (size of patch in pixels: strides+overlaps)
overlaps = (24, 24)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

mc_dict = {
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy'
}
opts = CNMFParams(params_dict=mc_dict)


def main(path):
    # adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )
    # Run MC
    fnames = [download_demo(str(path))]
    mc = MotionCorrect(fnames, dview = dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie = True)
    np.save(path + 'mc.npy', mc.mmap_file)



if __name__ == "__main__":
    main(sys.argv[1])