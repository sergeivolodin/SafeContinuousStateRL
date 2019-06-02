import numpy as np
from matplotlib import pyplot as plt
import os
from helpers import *
import argparse
from config import *

parser = argparse.ArgumentParser(description='Run all experiments in one setting')
parser.add_argument('--setting', type=str, help='Setting to use (small/big/...)')
parser.add_argument('--delay', type=float, help='Delay between commands')

# parsing arguments
args = parser.parse_args()

### Choosing a setting
setting = args.setting
settings = {'small': small_setting, 'medium': medium_setting}
assert setting in settings, "Please supply a valid setting, one of: " + str(settings.keys())
R = settings[setting]()

# name of the setting, common parameters, parameter groups, all parameters
setting_name, common, param_groups = R
parameters = [x for group in param_groups.values() for x in group]

# group -> what changes
varying = {group: varying_for_agent(param_groups[group]) for group in param_groups.keys()}

print('Which variables are changing for each agent?')
print(varying)

### Creating the `.sh` file
write_sh_file(setting_name, parameters, common, delay = args.delay)
