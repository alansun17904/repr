#!/bin/bash

for i in {1..40}; do
    for seed in {1..10}; do
        python3 src/reprs.py --case_id $i --model_id $seed --out_name src/data/reprs/var_case_${i}_model_${seed}.pkl
    done
done
