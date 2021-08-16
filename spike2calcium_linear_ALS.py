import numpy as np


def linear_(x, params):
    a, b = params
    return a*x+b


def spike2calcium(spike_times, ca_times, param):
    tau_r = param[0] # rise
    tau_d1 = param[1] # decay1
    tau_d2 = param[2] # decay2
    k = param[3]
    ca_trace = np.zeros(len(ca_times))
    for n, spk in enumerate(spike_times):
        ca_trace_tmp = np.exp(-(ca_times-spk)/tau_d1)*(1-np.exp(-(ca_times-spk)/tau_r))
        ca_trace_tmp[ca_times<=spk]=0
        ca_trace += ca_trace_tmp
        ca_trace_tmp = np.exp(-(ca_times-spk)/tau_d2)*(1-np.exp(-(ca_times-spk)/tau_r))
        ca_trace_tmp[ca_times<=spk]=0
        ca_trace += ca_trace_tmp*k
    return ca_trace


# def fit_spike2calcium_kernel(param_int, param_linear, ca_trace, spike_times, ca_times):
#     from scipy.optimize import minimize, Bounds
#     def mse_spike2calcium(param, param_linear=param_linear, ca_trace=ca_trace, spike_times=spike_times, ca_times=ca_times):
#         if isinstance(ca_trace, list):
#             err = 0
#             for n in range(len(ca_trace)):
#                 ca_trace_est = spike2calcium(spike_times[n], ca_times[n], param)
#                 # assuming different section has its own linear parameters
#                 ca_trace_est = linear_(ca_trace_est, param_linear[n])
#                 err=err+((ca_trace[n]-ca_trace_est)**2).sum()
#             return err
#         else:
#             ca_trace_est = spike2calcium(spike_times, ca_times, param)
#             ca_trace_est = linear_(ca_trace_est, param_linear)
#             return ((ca_trace-ca_trace_est)**2).sum()
    
#     options={'disp': 1, 'maxcor': 10, 'ftol': 2.220446049250313e-09, \
#              'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, \
#              'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
#     lb = np.zeros(len(param_int))
#     ub = np.zeros(len(param_int))
#     lb[0] = 1/10000 
#     ub[:] = np.inf
#     ub[0] = .2 # rise time should not be so long
#     bounds = Bounds(lb, ub)
#     res = minimize(mse_spike2calcium, param_int, method='L-BFGS-B', bounds=bounds, options=options)
#     return res.x


# decay only
def fit_spike2calcium_kernel(param_int, param_linear, ca_trace, spike_times, ca_times):
    from scipy.optimize import minimize, Bounds
    t_r = param_int[0]
    def mse_spike2calcium(param, t_r=t_r, param_linear=param_linear, ca_trace=ca_trace, spike_times=spike_times, ca_times=ca_times):
        param_ = np.r_[t_r, param]
        if isinstance(ca_trace, list):
            err = 0
            for n in range(len(ca_trace)):
                ca_trace_est = spike2calcium(spike_times[n], ca_times[n], param_)
                # assuming different section has its own linear parameters
                ca_trace_est = linear_(ca_trace_est, param_linear[n])
                err=err+((ca_trace[n]-ca_trace_est)**2).sum()
            return err
        else:
            ca_trace_est = spike2calcium(spike_times, ca_times, param_)
            ca_trace_est = linear_(ca_trace_est, param_linear)
            return ((ca_trace-ca_trace_est)**2).sum()
    
    options={'disp': 1, 'maxcor': 10, 'ftol': 2.220446049250313e-09, \
             'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, \
             'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    lb = np.zeros(len(param_int)-1)
    ub = np.zeros(len(param_int)-1)
    ub[:] = np.inf
    bounds = Bounds(lb, ub)
    res = minimize(mse_spike2calcium, param_int[1:], method='L-BFGS-B', bounds=bounds, options=options)
    return np.r_[t_r, res.x]


def fit_spike2calcium_linear(param_kernel, ca_trace, spike_times, ca_times):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    param_model = []
    if isinstance(ca_trace, list):
        for n in range(len(ca_trace)):
            ca_trace_est = spike2calcium(spike_times[n], ca_times[n], param_kernel)
            reg = model.fit(ca_trace_est[:, None], ca_trace[n])
            param_model.append(np.array([reg.coef_[0], reg.intercept_]))
    else:
        ca_trace_est = spike2calcium(spike_times, ca_times, param_kernel)
        reg = model.fit(ca_trace_est[:, None], ca_trace)
        param_model.append(np.array([reg.coef_[0], reg.intercept_]))
    return param_model


def ev_s2f(ca_est, dff):
    return 1-((ca_est-dff)**2).mean()/dff.var()