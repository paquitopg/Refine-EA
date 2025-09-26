# RefineEA Base Launcher

This launcher runs the RefineEA entity alignment pipeline using the VLLM server for LLM inference.

## Prerequisites

1. **VLLM Server Running**: Make sure the VLLM server is running (see `../server_launcher/`)
2. **Data Available**: Ensure your data is in the correct location

## Setup

1. **Copy environment file**:
   ```bash
   cp ".env copy" .env
   ```

2. **Edit `.env`** with your configuration:
   - `DATASET`: Dataset name (default: `airelle`)
   - `LLM_CONFIG`: Path to vLLM config (default: `configs/vllm_config.yaml`)
   - `NUM_CANDIDATES`: Number of candidates per entity (default: `10`)
   - `MAX_ENTITIES`: Maximum entities to process (default: `50`)
   - `OUTPUT_DIR`: Output directory (default: `outputs`)

## Usage

### Using SLURM (Recommended)
```bash
sbatch sbatch.sh
```

### Using Docker Compose
```bash
docker-compose run --rm refine_ea_service
```

## Configuration

The launcher uses the following configuration:

- **Data Directory**: `/raid/ml-data/paco/refine_ea_data/${DATASET}`
- **Output Directory**: `/home/paco/projects/Refine-EA/outputs/${DATASET}`
- **LLM Config**: `/home/paco/projects/Refine-EA/${LLM_CONFIG}`
- **Cache**: `/tmp/hf_cache_${USER_ID}`

## Resource Requirements

- **CPU**: 16 cores (configurable)
- **Memory**: Based on SLURM allocation
- **GPU**: **None required** - only makes HTTP requests to VLLM server
- **Storage**: Access to data and output directories

## Pipeline Flow

1. **Load Data**: Entity attributes from `KG1_entity_attributes.json` and `KG2_entity_attributes.json`
2. **Load Candidates**: Alignment candidates from `alignment_candidates.txt`
3. **Process Entities**: For each entity, get top N candidates
4. **LLM Reasoning**: Send prompts to VLLM server for entity matching
5. **Save Results**: Save alignment results and evaluation metrics

## Expected Output

- `alignment_results.json`: Entity alignment predictions
- `evaluation_metrics.json`: Performance metrics
- `refine_ea.log`: Pipeline logs 