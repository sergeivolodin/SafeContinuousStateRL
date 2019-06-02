import argparse
# one experiment!

import waitGPU
import os
import sys

# for environ
import os

# only using device 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='Run the experiment')
parser.add_argument('--agent', type=str, help='The agent to use')
parser.add_argument('--epochs', type=int, help='Maximal number of epochs')
parser.add_argument('--constraint', type=float, help='The constraint for the environment')
parser.add_argument('--episodes', type=int, help='Number of episodes to collect before training')
parser.add_argument('--repetitions', type=int, help='Number of repetitions of the experiment')

# agent args
parser.add_argument('--delta', type=float, help = "Delta (step size) for CPO")
parser.add_argument('--lr_policy', type=float, help = "sPPO")
parser.add_argument('--lr_value', type=float, help = "sPPO")
parser.add_argument('--lr_failsafe', type=float, help = "sPPO")
parser.add_argument('--steps', type=int, help = "sPPO")
parser.add_argument('--epsilon', type=float, help = "PPO")

# parsing arguments
args = parser.parse_args()

# FILENAME for output
params_describe = "_".join([x + "-" + str(y) for x, y in vars(args).items()]) + ".output"

# writing something there if it doesn't exist
# if exists, exiting
if os.path.isfile(params_describe):
  if open(params_describe, 'r').read() != 'Nothing[':
    print('Already exists')
    sys.exit(0)

# writing temporary data
open(params_describe, 'w').write('Nothing[')

# waiting for GPU
waitGPU.wait(nproc=4, interval = 10, gpu_ids = [0, 1])

from baselines import *
from saferl import *
from sppo import *

sess = create_modest_session()

if args.agent == 'sppo':
  params = ['epsilon', 'lr_policy', 'lr_value', 'lr_failsafe', 'steps']
  agent = ConstrainedProximalPolicyOptimization
elif args.agent == 'cpo':
  params = ['delta']
  agent = ConstrainedPolicyOptimization
elif args.agent == 'random':
  params = []
  agent = ConstrainedRandomAgent
else:
  print('Unknown agent ' + args.agent)
  sys.exit(0)

print('Creating environment')
# creating the environment...
env = make_safe_env('CartPole-v0-left-half')
env.threshold = args.constraint

def get_params_from_args(lst):
  """ Arguments from command line """
  return {x: vars(args)[x] for x in lst}

print('Creating the agent')
# train call
agent = agent(env, sess, **get_params_from_args(params))

print('Creting the train loop')
# creating the train loop...
loop = ConstrainedEpisodicTrainLoop(env, agent, episodes_to_collect=args.episodes)

# all results
results = []

# run for N repetitions
for i in range(args.repetitions):
    # initializing
    print('Training %d/%d' % (i, args.repetitions))
    sess.run(tf.global_variables_initializer())
    # reward is larger than possible to train for a fixed number of epochs
    # plotting every 500 steps
    r = loop.achieve_reward(1000, args.epochs, plot_every = 500, fig_name = "./figures/" + params_describe + ("_repetition-%d" % i) + ".pdf")
    results.append(r)

print('Writing results')
f = open(params_describe, "w")
f.write(str(results))
f.close()
