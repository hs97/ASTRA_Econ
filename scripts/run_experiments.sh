#!/bin/bash

# NAACL 2021 Experiments
baselogdir="$(date +'%h%d_%H-%M')_ECON_experiments"

for seed in 0; do
  logdir="${baselogdir}/seed${seed}"
  python3 ../astra/main.py --dataset econ --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data #--num_iter 1
  python3 ../astra/main.py --dataset econ_0 --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data #--num_iter 1
  python3 ../astra/main.py --dataset econ_mean --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data #--num_iter 1

done


# TREC
for seed in 0; do
  logdir="${baselogdir}/seed${seed}"
  python3 ../astra_reg/main.py --dataset econ_reg_EU --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data #--num_iter 1
  python3 ../astra_reg/main.py --dataset econ_reg_ffill --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data #--num_iter 1
  python3 ../astra_reg/main.py --dataset econ_reg --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data #--num_iter 1
done
