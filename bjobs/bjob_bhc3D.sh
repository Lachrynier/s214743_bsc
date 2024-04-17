#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J bhc3D
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:10
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000MB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o batch_output/bhc3D_%J.out
#BSUB -e batch_output/bhc3D_%J.err
# -- end of LSF options --

source ~/miniconda3/bin/activate
conda activate bsc
python -u ../bhc_3D.py