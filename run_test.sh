#!/bin/bash

set -v
ckpt=checkpoint_2021-04-25-23-24-30

i=49
total=50 # test/eval last chckpt
ckpt_dir=../topic-abstractive-sum/experiments2021/${ckpt}
mkdir -p ${ckpt_dir}/summary
while [[ $i -lt $total ]]
do
	fp="${ckpt_dir}/best-${i}.data-00000-of-00001"
	while [[ ! -f ${fp} ]]
	do
		echo "File doesn't exist: ${fp}"
		sleep 5m
	done
	echo Decoding using checkpoint: $fp
	python -u run.py --mode=test --init_checkpoint=${ckpt} --checkpoint_file=best-$i --num_gpus=1 --coverage=false --use_pointer=false &
	sleep 1
	srun -c 6 --gres=gpu:1 -J eval-news-$i -o ${ckpt_dir}/summary/eval.${i}.log  python -u run.py --mode=eval --init_checkpoint=${ckpt} --checkpoint_file=best-$i --num_gpus=1 --coverage=false --use_pointer=false &
	sleep 1s
	i=$[i+1]
done
