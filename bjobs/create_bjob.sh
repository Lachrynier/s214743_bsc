#!/bin/bash

# Customization variables
OUTPUT_DIR="batch_output"
FILE_NAME="bhc_v2"
TOTAL_RAM_MB=8000
NUM_CORES=4
RAM_PER_CORE=$(($TOTAL_RAM_MB / $NUM_CORES))

# Create the job file
cat > bjob_${FILE_NAME}.sh <<EOF
#!/bin/sh
### General options

### -- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J ${FILE_NAME}

### -- ask for number of cores (default: 1) --
#BSUB -n ${NUM_CORES}

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:05

# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${RAM_PER_CORE}MB]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ${OUTPUT_DIR}/${FILE_NAME}_%J.out
#BSUB -e ${OUTPUT_DIR}/${FILE_NAME}_%J.err
# -- end of LSF options --

source ~/miniconda3/bin/activate
conda activate bsc
python -u ../${FILE_NAME}.py
EOF