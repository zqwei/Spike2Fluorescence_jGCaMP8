import numpy as np


def linear_(x, params):
    a, b = params
    return a*x+b


def sigmoid(x, L, x0, k, b):
    return L/(1+np.exp(-k*(x-x0)))+b


def sigmoid_fit(x, y):
    from scipy.optimize import curve_fit
    ymin = np.percentile(y, 5)
    ymax = np.percentile(y, 95)
    x0 = np.percentile(x, 70)
    p0 = [ymax-ymin, x0, 1, ymin]
    popt, pcov = curve_fit(sigmoid, x, y, p0, method='lm', maxfev=1000)
    return np.array(popt)


def linear_fit(x, y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    reg = model.fit(x[:, None], y)
    a_tmp = reg.coef_[0]
    b_tmp = reg.intercept_
    return np.array([a_tmp, b_tmp, np.nan, np.nan])


# def sigmoid_fit_from_linear(x, y):
#     from scipy.optimize import curve_fit
#     from sklearn.linear_model import LinearRegression
#     model = LinearRegression(fit_intercept=True)
#     reg = model.fit(x[:, None], y)
#     a_tmp = reg.coef_[0]
#     b_tmp = reg.intercept_
#     k = .1
#     ymax = 4*a_tmp/k
#     ymin = b_tmp - ymax/2
#     x0 = 0
#     p0 = [ymax, x0, k, ymin]
#     popt, pcov = curve_fit(sigmoid, x, y, p0, method='lm', maxfev=1000)
#     return np.array(popt)


def spike2calcium(spike_times, ca_times, param):
    tau_r = param[0] # rise
    tau_d = param[1] # decay1
    ca_trace = np.zeros(len(ca_times))
    for n, spk in enumerate(spike_times):
        ca_trace_tmp = np.exp(-(ca_times-spk)/tau_d)*(1-np.exp(-(ca_times-spk)/tau_r))
        ca_trace_tmp[ca_times<=spk]=0
        ca_trace += ca_trace_tmp
    return ca_trace


def fit_spike2calcium_kernel(param_int, param_model, ca_trace, spike_times, ca_times):
    from scipy.optimize import minimize, Bounds
    def mse_spike2calcium(param, param_model=param_model, ca_trace=ca_trace, spike_times=spike_times, ca_times=ca_times):
        if isinstance(ca_trace, list):
            err = 0
            for n in range(len(ca_trace)):
                ca_trace_est = spike2calcium(spike_times[n], ca_times[n], param)
                # assuming different section has its own linear parameters
                if np.isnan(param_model[n][2]):
                    ca_trace_est = linear_(ca_trace_est, param_model[n][:2])
                else:
                    ca_trace_est = sigmoid(ca_trace_est, *param_model[n])
                err=err+((ca_trace[n]-ca_trace_est)**2).sum()
            return err
        else:
            ca_trace_est = spike2calcium(spike_times, ca_times, param)
            ca_trace_est = linear_(ca_trace_est, *param_model)
            return ((ca_trace-ca_trace_est)**2).sum()
    
    options={'disp': 1, 'maxcor': 10, 'ftol': 2.220446049250313e-09, \
             'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, \
             'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    lb = np.zeros(len(param_int))
    ub = np.zeros(len(param_int))
    lb[0] = 1/10000
    ub[:] = np.inf
    ub[0] = .2 # rise time should not be so long
    bounds = Bounds(lb, ub)
    res = minimize(mse_spike2calcium, param_int, method='L-BFGS-B', bounds=bounds, options=options)
    return res.x


def fit_spike2calcium_sigmoid(param_kernel, ca_trace, spike_times, ca_times):
    param_model = []
    if isinstance(ca_trace, list):
        for n in range(len(ca_trace)):
            ca_trace_est = spike2calcium(spike_times[n], ca_times[n], param_kernel)
            try:
                reg = sigmoid_fit(ca_trace_est, ca_trace[n])
            except:
                reg = linear_fit(ca_trace_est, ca_trace[n])
            param_model.append(reg)
    else:
        ca_trace_est = spike2calcium(spike_times, ca_times, param_kernel)
        reg = sigmoid_fit(ca_trace_est, ca_trace)
        param_model.append(reg)
    return param_model


def ev_s2f(ca_est, dff):
    return 1-((ca_est-dff)**2).mean()/dff.var()