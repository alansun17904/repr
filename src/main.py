#! /usr/bin/env python3
import pickle
import logging
import sys
from pathlib import Path

import jax

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.algorithms import run_algorithm
from circuits_benchmark.commands.train import train
from circuits_benchmark.commands.evaluation import evaluation

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
  parser = build_main_parser()

  args, _ = parser.parse_known_args(sys.argv[1:])


  # get the admissible cases
  admissible = pickle.load(open("admissible_tasks.pkl", "rb"))

  for case in admissible.values():
    print(f"Training against Case {case.get_name()}")
    for seed in range(0, 10):
      args.seed = seed
      args.indices= case.get_name()
      args.output_dir = str(Path(args.output_dir) / f"c{case.get_name()}-s{seed}")
      if args.command == "run":
        run_algorithm.run(args)
      elif args.command == "train":
        train.run(args)
      elif args.command == "eval":
        evaluation.run(args)
      sys.exit(1)