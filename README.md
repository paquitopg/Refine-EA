# RefinEA: Entity Alignment using LLM Reasoning

RefinEA is a modular system for entity alignment between knowledge graphs using Large Language Model (LLM) reasoning. The system takes entity attributes from two knowledge graphs and uses LLM-based reasoning to determine the best matches between entities.

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies**:
```bash
pip install torch transformers PyYAML accelerate bitsandbytes
```

2. **Prepare your data**:
   - Place your entity attributes in `data/your_dataset/`:
     - `KG1_entity_attributes.json`
     - `KG2_entity_attributes.json`
   - Include alignment files:
     - `alignment_candidates.txt`
     - `ref_pairs` (ground truth)

3. **Configure LLM**:
   - Edit `configs/llm_config.yaml` with your preferred model settings

### Basic Usage

```bash
python main.py \
    --data_dir data/airelle/ \
    --llm_config configs/llm_config.yaml \
    --num_candidates 10 \
    --output_dir outputs/airelle/
```

### Using Launcher Scripts

For the Airelle dataset:
```bash
bash launchers/run_airelle.sh
```

For any dataset:
```bash
bash launchers/run_cluster.sh data/your_dataset/ configs/llm_config.yaml 10 outputs/your_dataset/
```

## 📁 Data Structure

Your data directory should contain:

```
data/your_dataset/
├── KG1_entity_attributes.json    # KG1 entity attributes
├── KG2_entity_attributes.json    # KG2 entity attributes
├── alignment_candidates.txt      # Pre-computed candidates
└── ref_pairs                    # Ground truth alignments
```

### Entity Attributes Format

```json
{
  "0": {
    "type": "Company",
    "name": ["CNIM Group"],
    "description": ["A French global industrial engineering group"],
    "foundedYear": [1856],
    "keyStrengths": ["innovation", "technology"]
  }
}
```

### Alignment Candidates Format

```
# Format: kg1_entity_id	kg2_entity_id	similarity_score	rank
0	408	0.950553	1
0	363	0.590526	2
...
```

## 🔧 Configuration

### LLM Configuration (`configs/llm_config.yaml`)

```yaml
model:
  name: "meta-llama/Llama-2-13b-hf"
  type: "huggingface"
  device: "auto"
  load_in_8bit: true

generation:
  max_new_tokens: 1024
  temperature: 0.3
  top_p: 0.9
  do_sample: true
```

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--data_dir` | Yes | Path to data directory |
| `--llm_config` | Yes | Path to LLM config file |
| `--num_candidates` | Yes | Number of candidates per entity |
| `--output_dir` | No | Output directory (default: outputs) |
| `--max_entities` | No | Max entities to process |
| `--log_level` | No | Logging level (DEBUG/INFO/WARNING/ERROR) |

## 🏗️ Architecture

### Core Components

- **`EntityMatcher`**: LLM-based entity matching logic
- **`CandidateSelector`**: Manages alignment candidates
- **`AttributeExtractor`**: Handles entity attributes
- **`AlignmentPipeline`**: Main orchestration pipeline

### Data Flow

1. **Load Data**: Entity attributes and candidates
2. **Process Entities**: For each entity, get candidates and use LLM reasoning
3. **Evaluate Results**: Compare against ground truth
4. **Generate Reports**: Save results and metrics

## 📊 Output

The pipeline generates:

- **`alignment_results.json`**: Detailed alignment results
- **`evaluation_metrics.json`**: Performance metrics
- **`refinEA.log`**: Detailed execution log

### Evaluation Metrics

- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Average Confidence**: Mean confidence across all predictions

## 🔮 Advanced Usage

### Custom Entity Processing

```python
from refinEA.pipeline import AlignmentPipeline

pipeline = AlignmentPipeline(
    data_dir="data/your_dataset/",
    llm_config_path="configs/llm_config.yaml",
    num_candidates=15
)

# Process specific entities
results = pipeline.align_entities(entity_ids=["0", "1", "2"])

# Evaluate results
metrics = pipeline.evaluate_results(results)
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### Cluster Deployment

For cluster environments, use the launcher scripts:

```bash
# SLURM example
sbatch --job-name=refinEA \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=4 \
       --mem=32G \
       launchers/run_cluster.sh \
       data/airelle/ \
       configs/llm_config.yaml \
       10 \
       outputs/airelle/
```

## 🛠️ Development

### Project Structure

```
refinEA/
├── main.py                    # Main entry point
├── refinEA/
│   ├── llm/                  # LLM integration
│   ├── matching/             # Entity matching
│   ├── pipeline/             # Pipeline orchestration
│   └── utils/                # Utilities
├── configs/                  # Configuration files
├── data/                     # Data directories
├── outputs/                  # Output directories
├── launchers/               # Cluster launcher scripts
└── examples/                # Example scripts
```

### Adding New Features

1. **New LLM Backend**: Extend `BaseLLMInterface`
2. **New Matching Strategy**: Add to `matching/` module
3. **New Evaluation Metric**: Extend `AlignmentPipeline.evaluate_results()`

## 📝 Notes

- All paths are configurable via command-line arguments
- The system is designed for cluster deployment
- Comprehensive logging and error handling
- Modular architecture for easy extension

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. 