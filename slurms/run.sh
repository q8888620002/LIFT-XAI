#!/bin/bash

# let's source our ~/.bashrc for 'conda', then activate our environment.
. ~/.bashrc
conda activate lift_xai

export PYTHONPATH="$PYTHONPATH:$PWD"
export XDG_CACHE_HOME="/gscratch/efml/mingyulu/.cache"

eval $*