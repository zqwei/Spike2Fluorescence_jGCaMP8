from spike2calcium_single_decay import *
import os, sys
import warnings
warnings.filterwarnings('ignore')
from glob import glob

# get cell list information
flist = sorted(glob('GCaMP8_exported_ROIs_s2f_full/dff/*.npz'))
cell_info=["_".join(os.path.basename(f).split('_')[:6])[:-4] for f in flist]
cell_info_uq = np.unique(cell_info)

# set constants for parameter fits
model_list = [linear_, quadratic_, hill_, sigmoid_]
model_name = ['linear', 'quadratic', 'hill function', 'sigmoid']

# fitting codes
def run_single_cell(nf, model_list=model_list, model_name=model_name):
    nf = nf.squeeze()
    _ = np.load(f'GCaMP8_exported_ROIs_s2f_full/dff/{nf}.npz', allow_pickle=True)
    spike_times = _['spike_times'] # raw fluorescence of the ROI
    ca_times = _['ca_times'] # raw fluorescence of the surrounding neuropil, for neuropil subtraction
    ca_trace = _['ca_trace'] # frame rate of the movie
    cell_type = _['cell_type'] # timing of the action potentials from ephys
    valid_trial = np.array([len(st) for st in spike_times])>=12
    
    spike_times = [_ for _ in spike_times[valid_trial]]
    ca_times = [_ for _ in ca_times[valid_trial]]
    ca_trace =[_ for _ in ca_trace[valid_trial]]
    
    param_est = []
    if os.path.exists('GCaMP8_exported_ROIs_s2f_full/fit_sd_results_v00/'+nf+'_fit_v00.npz'):
        _=np.load('GCaMP8_exported_ROIs_s2f_full/fit_sd_results_v00/'+nf+'_fit_v00.npz', allow_pickle=True)
        param_int_list = _['param_est']
    
    for model, param_int in zip(model_list, param_int_list):
        param = fit_spike2calcium(param_int, ca_trace, spike_times, ca_times, model)
        print(f'finished: {model}')
        param_est.append(param)
    np.savez('GCaMP8_exported_ROIs_s2f_full/fit_sd_results_v01/'+nf+'_fit', param_est=np.array(param_est))
    return None


if __name__ == "__main__":
    import dask
    res = [dask.delayed(run_single_cell)(nf) for nf in cell_info_uq]
    dask.compute(res)
