#!/bin/bash
pids=""
python ../experiment.py --agent cpo --delta 0.05 --epochs 100 --constraint 100.0 --episodes 1 --repetitions 2 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent random --epochs 100 --constraint 100.0 --episodes 1 --repetitions 2 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 100 --constraint 100.0 --episodes 1 --repetitions 2 &
pids="$pids $!"
sleep 5.00
wait $pids
