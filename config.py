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
