#!/bin/bash


python src/reprs.py src/data/constant-ll-models src/reprs/constant-ll-models --eval
python src/reprs.py src/data/corr-ll-models src/reprs/corr-ll-models --eval
python src/reprs.py src/data/corr-arch-ll-models src/reprs/corr-arch-ll-models --eval
python src/reprs.py src/data/arch-ll-models src/reprs/arch-ll-models --eval


python src/eval.py