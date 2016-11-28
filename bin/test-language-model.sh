#!/bin/sh

set -xe

PYTHONPATH=$(pwd)
export PYTHONPATH

if [ ! -f DeepSpeech.ipynb ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python util/lm/wiki_word_lm.py --use_checkpoint True
