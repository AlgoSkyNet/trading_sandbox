import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from zipline import TradingAlgorithm

from joblib import Parallel, delayed
import os

def run_algo_single(**algo_descr):
    if 'constraint_func' in algo_descr:
        if algo_descr['constraint_func'](algo_descr['param_set']):
            return np.nan

    try:
        algo = TradingAlgorithm(initialize=algo_descr['initialize'],
                                handle_data=algo_descr['handle_data'],
                                **algo_descr['param_set']
        )
        perf = algo.run(algo_descr['data'])
        daily_rets = perf.portfolio_value.pct_change().dropna()
        
        if daily_rets.std() > 0:
            sharpe_ratio_calc = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
        else:
            sharpe_ratio_calc = -999

        risk_report = algo.risk_report
        risk_cum = pd.Series(algo.perf_tracker.cumulative_risk_metrics.to_dict())
    except ImportError as e:
        print(e)
        return np.nan

    # Apply objective functions
    objective = algo_descr.get('objective', 'none')
    if objective == 'none':
        obj = (perf, risk_cum, risk_report)
    elif objective == 'sharpe':
        obj = sharpe_ratio_calc
        
    elif objective == 'total_return':
        obj = perf['portfolio_value'][-1] / perf['portfolio_value'][0] - 1
    elif callable(objective):
        obj = objective(perf, risk_cum, risk_report)
    else:
        raise NotImplemented('Objective %s not implemented.' % algo_descr['objective'])

    print "Sharpe: " + str(sharpe_ratio_calc) + "  %_Return: " + str(perf.portfolio_value[-1]/perf.portfolio_value[0]-1) + "  MaxDD: " + str(perf.max_drawdown[-1]) + "  MaxExp: " + str(perf.max_leverage[-1])

    return obj

def run_algo(*params, **algo_descr):
    if len(params) != 0:
        #algo_descr['param_set'] = pd.Series(params, index=algo_descr['params'].keys()).to_dict()
        algo_descr['param_set'] = params[0]

    if algo_descr.get('crossval', 'single') == 'single':
        obj = run_algo_single(**algo_descr)
    elif algo_descr['crossval'] == 'bootstrap': # Run bootstrap
        data = algo_descr['data'].copy()
        objs = []
        # Run over one year each
        for dt, sub_data in data.groupby(data.major_axis.year):
            algo_descr['data'] = sub_data
            objs.append(run_algo_single(**algo_descr))
        obj = np.mean(objs)
    else:
        raise NotImplementedError("Crossvalidation method %s not understood. Use 'single' or 'bootstrap'." % algo['crossval'])

    print("done", params, obj)
    return obj

def run_algo_joblib_wrapper(param_set, name, objective, initialize, handle_data, data):
    obj = run_algo_single(
        name=name,
        objective=objective,
        initialize=initialize,
        handle_data=handle_data,
        param_set=param_set,
        data=data,
        )

    return obj

def run_grid_search(params, name, objective, initialize, handle_data, data):
    spaces = []
    for param in params.itervalues():
        if param['type'] == 'int':
            space = np.append(np.arange(param['min'], param['max'], param['steps']), [param['max']])
        elif param['type'] == 'float':
            space = np.linspace(param['min'], param['max'], param['steps'])
        else:
            raise NotImplementedError('Type %s not implemented.' % type)

        spaces.append(space)

    grid = np.meshgrid(*spaces)
    # Convert grid so that it's iterable
    #grid = pd.DataFrame(grid.reshape((grid.shape[0], -1)).T,
    #                    columns=algo['params'].keys())
    #grid_obj = Parallel(n_jobs=10)(run_algo_vec(*grid, **algo_descr))
    xv, yv = grid
    it_grid = []
    for xi in spaces[0]:
        for yi in spaces[1]:
            it_grid.append({
                'fast_ma': xi,
                'slow_ma': yi,
                    })
    print(it_grid)
    os.system("taskset -p 0xffffffff %d" % os.getpid()) # numpy (openBLAS) breaks this in ubuntu for joblib
    grid_obj = Parallel(n_jobs=-2)(delayed(run_algo_joblib_wrapper)(g_p, name, objective, initialize, handle_data, data) for g_p in it_grid)
    print(grid_obj)
    grid.append(grid_obj)
    results = pd.DataFrame(np.asarray([np.ravel(x) for x in grid]).T,
                           columns=params.keys() + ['objective'])
    return results

def run_sigopt(samples=100, experiment_id=None, user_token=None, client_token=None, client_id=None, **algo_descr):
    import sigopt.interface

    if user_token is None or client_token is None or client_id is None:
        raise ValueError('No sigopt credentials passed, find them at https://sigopt.com/user/profile')

    conn = sigopt.interface.Connection(user_token=user_token, client_token=client_token)

    if experiment_id != None:
        resp = conn.clients(client_id).experiments()
        found = False
        for exp in resp.experiments:
            if int(exp.id) == experiment_id:
                experiment = exp
                found = True
                break
        if found == False:
            raise ValueError("Experiment id {0} not found for client id {1}".format(experiment_id, client_id))
    else:
        experiment = conn.experiments.create(client_id=client_id, data={
            'name': 'Quantopian_POC_talib_1',
            'parameters': [{
                        'name': 'fast_ma',
                        'type': 'int',
                        'bounds': { 'min': 5, 'max': 100 }},
                           {
                        'name': 'slow_ma',
                        'type': 'int',
                        'bounds': { 'min': 5, 'max': 300 }},
                          ],
            }).experiment

    for trial in range(samples):
        print("running trial: {0}".format(trial))
        suggestion = conn.experiments(experiment.id).suggest().suggestion

        # Get the Sharpe Ratio
        algo_descr['param_set'] = suggestion.assignments.to_json()
        print(algo_descr['param_set'])
        obj = run_algo(**algo_descr)

        conn.experiments(experiment.id).report(data={
          'assignments': suggestion.assignments,
          'value': obj,
        })

    exp_hist = conn.experiments(experiment.id).history()

    exp_data = exp_hist.observations.to_json()['data']
    exp_data.reverse() # start at the beginning

    # load up dataframe
    from collections import defaultdict
    sigopt_df_dict = defaultdict(list)
    for data_point in exp_data:
        for k, v in data_point['assignments'].iteritems():
            sigopt_df_dict[k].append(v)
        sigopt_df_dict['objective'].append(data_point['value'])

    sigopt_df = pd.DataFrame(sigopt_df_dict)

    return sigopt_df

def plot_max_over_time(x, line_kws=None, point_kws=None):
    default_line_kws = {'color': 'g'}
    default_point_kws = {'color': 'b', 'marker': 'o', 'linestyle': ''}

    if line_kws is None:
        line_kws = {}
    if point_kws is None:
        point_kws = {}

    default_line_kws.update(line_kws)
    default_point_kws.update(point_kws)

    cummax = np.maximum.accumulate(x)
    plt.plot(cummax, **default_line_kws)
    plt.plot(x, **default_point_kws)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
