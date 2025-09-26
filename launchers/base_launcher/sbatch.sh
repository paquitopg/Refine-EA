#!/bin/bash
#SBATCH --job-name=refine_ea_alignment
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16


### Variables not to be touched ###
REFINE_EA_CONTAINER_NAME=refine_ea_${USER}_$SLURM_JOBID
CPU_IDS=$(scontrol show job -d $SLURM_JOBID | grep -o "CPU_IDs=[^ ]*" | cut -d= -f2-)
MEMORY=$(scontrol show job -d $SLURM_JOBID | grep -Eo "mem=[0-9]{1,4}*" | cut -d= -f2-)
DOCKER_RAM=$(echo $MEMORY | cut -d ' ' -f 1)g
USER_UID=$(id -u ${USER})
USER_GID=$(id -g ${USER})
DOCKER_GID=$(getent group docker | cut -d: -f3)
DOCKER_UID="${USER_UID}:${USER_GID}"
### ----------------------------- ###

# Get user and group IDs
USER_ID=$(id -u ${USER})
USER_GID=$(id -g ${USER})
DOCKER_GID=$(getent group docker | cut -d: -f3)
DGX_OPEN_PORT=666

# Create user-specific cache directory
mkdir -p /tmp/hf_cache_${USER_ID}

REFINE_EA_CONTAINER_NAME=$REFINE_EA_CONTAINER_NAME \
USER_UID=$USER_ID \
USER_GID=$USER_GID \
DOCKER_GID=$DOCKER_GID \
DOCKER_RAM=$DOCKER_RAM \
DGX_OPEN_PORT=$DGX_OPEN_PORT \
CPU_IDS=$CPU_IDS \
docker-compose run --rm refine_ea_service