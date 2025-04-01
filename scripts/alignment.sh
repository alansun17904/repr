#!/bin/bash

python3 src/alignment.py src/data/reprs/constant-ll-models src/data/reprs/constant-ll-models/alignment.pkl
python3 src/alignment.py src/data/reprs/corr-ll-models src/data/reprs/corr-ll-models/alignment.pkl
python3 src/alignment.py src/data/reprs/arch-ll-models src/data/reprs/arch-ll-models/alignment.pkl
python3 src/alignment.py src/data/reprs/corr-arch-ll-models src/data/reprs/corr-arch-ll-models/alignment.pkl