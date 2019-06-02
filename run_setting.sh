#!/bin/bash
if [ "X$1" == "X" ]
then
	echo "Usage: $0 setting_name (big/small/...) [delay_s]"
	exit 1
fi

if [ "X$2" == "X" ]
then
  delay=5
else
  delay=$2
fi

python create_run.py --setting $1 || exit 1
out_fn=$(python create_run.py --delay $delay --setting $1|grep OUTPUT|cut -d " " -f 2)
cd output
bash ../$out_fn
cd ..
python analyze_run.py --setting $1 | tee run_${1}.txt
