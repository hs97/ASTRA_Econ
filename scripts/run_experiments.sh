#!/bin/bash

# NAACL 2021 Experiments
baselogdir="$(date +'%h%d_%H-%M')_ECON_experiments"

#for seed in 0; do
#  logdir="${baselogdir}/seed${seed}"
#  python3 ../astra/main.py --dataset econ --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data
#done


# TREC
for seed in 0; do
  logdir="${baselogdir}/seed${seed}"
  python3 ../astra_reg/main.py --dataset econ_reg --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data
done
