import numpy as np

def small_setting():
    # SMALL setting experiments
    setting_name = 'small'
    common = {'epochs': 100, 'constraint': 100.0, 'episodes': 1, 'repetitions': 2}
    param_groups = {
        'cpo':
        [{'agent': 'cpo', 'delta': 0.05, **common}],

        'random':
        [{'agent': 'random', **common}],

        'sppo':
        [{'agent': 'sppo', 'epsilon': 0.1, 'steps': 5, 'lr_policy': 1e-3, 'lr_value': 1e-3, 'lr_failsafe': 1e-3, **common}],
    }
    return setting_name, common, param_groups

def medium_setting():
    # SMALL setting experiments
    setting_name = 'medium'
    exp_search = [0.1, 0.001]
    common = {'epochs': 5000, 'constraint': 100.0, 'episodes': 5, 'repetitions': 5}
    param_groups = {
        'cpo':
        [{'agent': 'cpo', 'delta': delta, **common} for delta in [0.1, 0.5, 0.05, 0.01, 0.005, 0.001]],

        'random':
        [{'agent': 'random', **common}],

        'sppo':
        [{'agent': 'sppo', 'epsilon': eps, 'steps': steps, 'lr_policy': lr_policy, 'lr_value': lr_value, 'lr_failsafe': lr_failsafe, **common} for eps in [0.1] for steps in [1, 5, 10, 20] for lr_value in exp_search for lr_policy in exp_search for lr_failsafe in exp_search],
    }
    return setting_name, common, param_groups
