'''
From fit v01, remove the dff trace with related low snr
'''
from dff import *
import os, sys
import warnings
warnings.filterwarnings('ignore')
from glob import glob

# get cell list information
flist = sorted(glob('GCaMP8_exported_ROIs_s2f_full/data/*.npz'))
cell_info=["_".join(os.path.basename(f).split('_')[:6]) for f in flist]
cell_info_uq = np.unique(cell_info)

bname = 'GCaMP8_exported_ROIs_s2f_full/dff/'

# fitting codes
def run_single_cell(nf, bname=bname):
    nf = nf.squeeze()
    files = sorted(glob(f'GCaMP8_exported_ROIs_s2f_full/data/{nf}*.npz'))
    cell_type = files[0][-7:][:3]
    ca_trace = []
    spike_times = []
    ca_times = []
    ev_linear_est = []
    cell_type
    for nf_ in files:
        _ = np.load(nf_, allow_pickle=True)
        F = _['F'] # raw fluorescence of the ROI
        Fneu = _['Fneu'] # raw fluorescence of the surrounding neuropil, for neuropil subtraction
        framerate = _['framerate'] # frame rate of the movie
        ap_times = _['ap_times'] # timing of the action potentials from ephys
        ap_times = np.array([float(tmp) for tmp in ap_times])
        frame_times = _['frame_times'] # timing of each frame
        neuropil_r = 0.8
        fmean_comp, baseline, dff = get_baseline_dff(F, Fneu, cont_ratio=neuropil_r, win_=1000, q=0.15)
        ca_trace.append(dff) # dff
        spike_times.append(ap_times) # spike time
        ca_times.append(frame_times) # frame_times
    np.savez(bname+nf, spike_times=np.array(spike_times), ca_times=np.array(ca_times), ca_trace=np.array(ca_trace), cell_type=cell_type)
    return None


if __name__ == "__main__":
    import dask
    res = [dask.delayed(run_single_cell)(nf) for nf in cell_info_uq]
    dask.compute(res)
