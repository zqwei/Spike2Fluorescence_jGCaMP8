from spike2calcium_sigmoid_ALS import *
import os, sys
import warnings
warnings.filterwarnings('ignore')
from glob import glob

# get cell list information
# flist = sorted(glob('GCaMP8_exported_ROIs_s2f_full/dff_v00/*.npz'))
# new fit based on the clean-up data
flist = sorted(glob('GCaMP8_exported_ROIs_s2f_full/dff_v03/*.npz'))
cell_info=["_".join(os.path.basename(f).split('_')[:6])[:-4] for f in flist]
cell_info_uq = np.unique(cell_info)


# fitting codes
def run_single_cell(nf):
    nf = nf.squeeze()
    _ = np.load(f'GCaMP8_exported_ROIs_s2f_full/dff_v03/{nf}.npz', allow_pickle=True)
    spike_times = _['spike_times'] # raw fluorescence of the ROI
    ca_times = _['ca_times'] # raw fluorescence of the surrounding neuropil, for neuropil subtraction
    ca_trace = _['ca_trace'] # frame rate of the movie
    cell_type = _['cell_type'] # timing of the action potentials from ephys
    spike_times = [_ for _ in spike_times]
    ca_times = [_ for _ in ca_times]
    ca_trace =[_ for _ in ca_trace]
    
    _=np.load('GCaMP8_exported_ROIs_s2f_full/fit_als_linear_results_v06/'+nf+'_fit.npz', allow_pickle=True)
    param_kernel = _['param_kernel']
    if param_kernel[3]==0:
        # using single decay fit instead here
        _=np.load('GCaMP8_exported_ROIs_s2f_full/fit_als_sigmoid_sd_results_v04/'+nf+'_fit.npz', allow_pickle=True)
        param_kernel = _['param_kernel']
        param_model = _['param_model']
        np.savez('GCaMP8_exported_ROIs_s2f_full/fit_als_sigmoid_results_v02/'+nf+'_fit', \
                 param_model=np.array(param_model), param_kernel=np.array(param_kernel))
        return None
    
    for n in range(3): # do 4 times of ALS at 1st round, 2 times in 2nd
        param_model = fit_spike2calcium_sigmoid(param_kernel, ca_trace, spike_times, ca_times)
        param_kernel = fit_spike2calcium_kernel(param_kernel, param_model, ca_trace, spike_times, ca_times)
    param_model = fit_spike2calcium_sigmoid(param_kernel, ca_trace, spike_times, ca_times)
    
    np.savez('GCaMP8_exported_ROIs_s2f_full/fit_als_sigmoid_results_v02/'+nf+'_fit', param_model=np.array(param_model), param_kernel=np.array(param_kernel))
    return None


if __name__ == "__main__":
    import dask
    res = [dask.delayed(run_single_cell)(nf) for nf in cell_info_uq]
    dask.compute(res)
