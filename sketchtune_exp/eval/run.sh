#!/usr/bin/env bash
set -eu

#
# this is a wrapper script to run the inference
# Participants are expected to change this script to suit their model

langs=$1
batch_size=$2

mydir=$(dirname "$0")
mydir=$(realpath "$mydir")

python -m sketchtune_exp.eval.run $langs $batch_size -m $mydir


