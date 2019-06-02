#!/bin/bash
pids=""
python ../experiment.py --agent cpo --delta 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent cpo --delta 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent random --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 1 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 2 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 3 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 4 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 6 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 10 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.1 --steps 20 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 1 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 2 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 3 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 4 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 6 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 10 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.001 --steps 20 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 1 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 2 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 3 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 4 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 5 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 6 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
wait $pids
sleep 5.00
pids=""
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 10 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.1 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.001 --lr_value 0.1 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.1 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.1 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
python ../experiment.py --agent sppo --epsilon 0.0001 --steps 20 --lr_policy 0.001 --lr_value 0.001 --lr_failsafe 0.001 --epochs 5000 --constraint 100.0 --episodes 5 --repetitions 5 &
pids="$pids $!"
sleep 5.00
wait $pids
