#!/bin/bash

# train
python -u run.py --num_gpus=1 --bert_config_file=./bert/sample/bert_config.json

# test
python -u run.py --mode=test --init_checkpoint=checkpoint_2022-01-20-16-21-22 --checkpoint_file=best-0 --num_gpus=1 --coverage=false --use_pointer=false
# eval
python -u run.py --mode=eval --init_checkpoint=checkpoint_2022-01-20-16-21-22 --checkpoint_file=best-0 --num_gpus=1 --coverage=false --use_pointer=false

