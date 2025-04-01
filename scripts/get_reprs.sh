#!/bin/bash

python src/reprs.py src/data/constant-ll-models src/data/reprs/constant-ll-models
python src/reprs.py src/data/corr-ll-models src/data/reprs/corr-ll-models
python src/reprs.py src/data/arch-ll-models src/data/reprs/arch-ll-models
python src/reprs.py src/data/corr-arch-ll-models src/data/reprs/corr-arch-ll-models
