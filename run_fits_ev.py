from spike2calcium import *
import os, sys
import warnings
warnings.filterwarnings('ignore')
from glob import glob

# get cell list information
flist = sorted(glob('GCaMP8_exported_ROIs_s2f_full/dff/*.npz'))
cell_info=["_".join(os.path.basename(f).split('_')[:6]) for f in flist]
cell_info_uq = np.unique(cell_info)

# set constants for parameter fits
model_list = [linear_, quadratic_, hill_, sigmoid_]
model_name = ['linear', 'quadratic', 'hill function', 'sigmoid']
bname = 'GCaMP8_exported_ROIs_s2f_full/fit_results/'
if not os.path.exists(bname):
    os.mkdir(bname)

# fitting codes
def run_single_cell(nf, model_list=model_list, model_name=model_name, bname=bname):
    nf = nf.squeeze()
    _ = np.load(f'GCaMP8_exported_ROIs_s2f_full/dff/{nf}*.npz'), allow_pickle=True)
    spike_times = _['spike_times'] # raw fluorescence of the ROI
    ca_times = _['ca_times'] # raw fluorescence of the surrounding neuropil, for neuropil subtraction
    ca_trace = _['ca_trace'] # frame rate of the movie
    cell_type = _['cell_type'] # timing of the action potentials from ephys
    valid_trial = np.array([len(st) for st in spike_times])>=12
    
    spike_times = spike_times[valid_trial]
    ca_times = ca_times[valid_trial]
    ca_trace = ca_trace[valid_trial]
    
    ca_est = []
    param_est = []
    ev = []
    
    if os.path.exists(bname+nf+'_fit_v00.npz'):
        _=np.load(bname+nf+'_fit_v00.npz', allow_pickle=True)
        param_int_list = _['param_est']
    
    for model, param_int in zip(model_list, param_int_list):
        param = fit_spike2calcium(param_int, ca_trace, spike_times, ca_times, model)
        print(f'finished: {model}')
        param_est.append(param)
        for nfits in range(len(ca_trace)):
            ca_est_ = spike2calcium(spike_times[nfits], ca_times[nfits], param, model)
            ca_est.append(ca_est_)
            ev.append(ev_s2f(ca_est_, ca_trace[nfits]))
    np.savez(bname+nf+'_fit_v00', param_est=np.array(param_est), ca_est=np.array(ca_est), dff=np.array(ca_trace), ev=np.array(ev))
    return None


if __name__ == "__main__":
    import dask
    res = [dask.delayed(run_single_cell)(nf) for nf in cell_info_uq]
    dask.compute(res)
