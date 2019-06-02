import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys, os

def restore_font():
  """ Restore matplotlib font """
  matplotlib.rc('font', size = 10)

def set_big_font():
  """ Set big font size in matplotlib """
  font = {'weight' : 'normal',
      'size'   : 20}

  matplotlib.rc('font', **font)

# folder with .sh scripts and .output files
output_folder = "./output/"
figures_folder = "./output/figures/"

def varying_for_agent(d):
    """ What changes for optimizer? """
    d0 = d[0]
    keys = set()
    for v in d:
        for key, val in v.items():
            if d0[key] != val:
                keys.add(key)
    return list(keys)

def print_nice(y):
    """ Print as a float/other """
    if isinstance(y, float):
        return str(round(y, 10))#'%.2g' % y
    return str(y)

def print_one(**kwargs):
    """ Print run info from kwargs """
    return('python ../experiment.py ' + " ".join(['--' + x + ' ' + print_nice(y) for x, y in kwargs.items()]) + ' &')

def args_in_order_():
    """ Arguments in the order experiment.py expects them """
    # arguments in the correct order
    f = open('experiment.py', 'r').readlines()
    args_in_order = []
    for l in f:
        k = 'parser.add_argument(\'--'
        if l.startswith(k):
            args_in_order.append(l[len(k):].split('\'')[0])
    return args_in_order

def get_file(**kwargs):
    """ Get output filename from kwargs """
    return (output_folder + "_".join([x + '-' + print_nice(kwargs[x] if x in kwargs else None) for x in args_in_order_()])+'.output')


def write_sh_file(setting_name, parameters, common, delay = None):
    """ Create .sh file with current setting """
    fn = output_folder + 'run_' + setting_name + '.sh'
    out = open(fn, 'w')

    if delay is None: delay = 5

    def write_to_out(s):
        #print(s)
        out.write(s + '\n')

    it = 0
    write_to_out('#!/bin/bash')
    for params in parameters:
        if it % 4 == 0:
            write_to_out('pids=""')
        write_to_out(print_one(**params))
        #print('echo aba; sleep 3 &')
        write_to_out('pids="$pids $!"')

        if it % 4 == 3:
            write_to_out('wait $pids')
        write_to_out('sleep %.2f' % delay)
        it += 1
    write_to_out('wait $pids')
    it = len(parameters)
    print('Total train stages: ', it * common['repetitions'])
    print('Total time, hours (approx): ', common['repetitions'] * 5 * it / 4 / 60)
    print('OUTPUT: ' + fn)

    out.close()

def arr_of_dicts_to_dict_of_arrays(arr):
    """ Array of dicts to dict of arrays """
    all_keys = arr[0].keys()
    return {key: [v[key] for v in arr] for key in all_keys}

def shorten_name(n):
    """ Shorten aba_caba to a_c """
    return '_'.join([x[:2] if len(x) else '' for x in n.split('_')])


def shorten_dict(d, filename = False):
    """ Shorten dictionary into a string """
    if filename:
        return '_'.join([shorten_name(x) + '-' + str(y) for x, y in d.items()])
    if len(d) == 1:
        return list(d.values())[0]
    return ', '.join([shorten_name(x) + ': ' + str(y) for x, y in d.items()])


def arr_to_stat(arr):
    """ Array -> mean, std """
    return (np.mean(arr), np.std(arr))

def dict_to_stat(d):
    if d is None:
        return None
    """ Dict key-> arr TO key -> mean, std"""
    return {x: arr_to_stat(y) for x, y in d.items()}

def dict_select(d, ks):
    """ Select only keys from ks from dict d """
    return {x: d[x] for x in ks}

def subplots(n, m, name, fcn, figsize = (10, 13), figname = None):
    """ Plot many subplots Width m, Height n, fcn: thing to plot """
    fig, axs = plt.subplots(n, m, figsize=figsize)
    axs = axs.ravel()
    i = 0
    while i < len(name):
        fcn(axs[i], i)
        axs[i].set_title(shorten_name(name[i]))
        if i == len(name) - 1 or i == len(name) - 2:
            plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=90)
        else:
            axs[i].set_xticks([])
        i += 1
    while i < n * m:
        fig.delaxes(axs[i])
        i += 1
    plt.show()
    if figname is not None:
        out_fn = figures_folder + figname + '.pdf'
        print("Output figure for results: " + out_fn)
        fig.savefig(out_fn, bbox_inches = 'tight')
    return fig

