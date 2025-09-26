#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --nodes=1
#SBATCH --gpus=4g.40gb:1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

### Variables not to be touched ###
SERVER_REFINE_EA_CONTAINER_NAME=vllm-server-${USER}-$SLURM_JOBID
CPU_IDS=$(scontrol show job -d $SLURM_JOBID | grep -o "CPU_IDs=[^ ]*" | cut -d= -f2-)
MEMORY=$(scontrol show job -d $SLURM_JOBID | grep -Eo "mem=[0-9]{1,4}*" | cut -d= -f2-)
DOCKER_RAM=$(echo $MEMORY | cut -d ' ' -f 1)g
USER_UID=$(id -u ${USER})
USER_GID=$(id -g ${USER})
DOCKER_GID=$(getent group docker | cut -d: -f3)
DOCKER_UID="${USER_UID}:${USER_GID}"
GPU_MIG_UUIDS=$(nvidia-smi -L | grep -o "UUID:[^)]*" | cut -d: -f2)
DOCKER_GPU=$(echo "$GPU_MIG_UUIDS" | tr " " ",")
### ----------------------------- ###


### Variable to be modified ###
DGX_OPEN_PORT=1444 # The port for Vllm to communicate with the container.
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # The model name to be loaded by Vllm.

# Example optional arguments to pass to the vllm command.
VLLM_ARGS="--gpu-memory-utilization 0.98 --enforce-eager"

# Load HF_TOKEN from .env file if it exists
if [ -f .env ]; then
    # Convert Windows line endings to Unix and source the .env file
    tr -d '\r' < .env > .env.tmp && source .env.tmp && rm .env.tmp
fi
### ----------------------------- ###

# Pass the env variables to the docker compose like this. It's not the most elegant way but it works.
echo "🐳 Starting docker-compose..."
echo "📋 Using model: $MODEL_NAME"
echo "📋 Using args: $VLLM_ARGS"

VLLM_ARGS="$VLLM_ARGS" \
SERVER_REFINE_EA_CONTAINER_NAME="$SERVER_REFINE_EA_CONTAINER_NAME" \
DOCKER_RAM="$DOCKER_RAM" \
DOCKER_GPU="$DOCKER_GPU" \
DOCKER_GID="$DOCKER_GID" \
DOCKER_UID="$DOCKER_UID" \
USER_UID="$USER_UID" \
CPU_IDS="$CPU_IDS" \
DGX_OPEN_PORT="$DGX_OPEN_PORT" \
MODEL_NAME="$MODEL_NAME" \
HF_TOKEN="$HF_TOKEN" \
docker-compose up