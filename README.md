# RefinEA: Entity Alignment using LLM Reasoning

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/paquitopg/Refine-EA)](https://github.com/paquitopg/Refine-EA)

RefinEA is a modular system for entity alignment between knowledge graphs using Large Language Model (LLM) reasoning. The system takes entity attributes from two knowledge graphs and uses LLM-based reasoning to determine the best matches between entities, performing discrimination between alignment candidates.

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

### Ground Truth Format

```
# Format: kg1_entity_id	kg2_entity_id
206	543
249	583
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

prompts:
  entity_matching: |
    You are an AI assistant that helps match entities...
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

### High-Level Architecture

RefinEA is designed as a modular, extensible system for entity alignment between knowledge graphs using LLM reasoning. The architecture follows a pipeline pattern with clear separation of concerns.

### Core Components

```
refinEA/
├── llm/                    # LLM integration layer
│   ├── base.py            # Abstract base classes
│   ├── huggingface_interface.py  # HuggingFace implementation
│   └── entity_matcher.py  # Entity matching logic
├── matching/              # Entity matching components
│   ├── entity_matcher.py  # Main matching logic
│   ├── candidate_selector.py  # Candidate management
│   └── attribute_extractor.py  # Attribute extraction
├── pipeline/              # Pipeline orchestration
│   └── alignment_pipeline.py  # Main pipeline
├── utils/                 # Utilities
│   └── config_loader.py   # Configuration management
└── examples/              # Example scripts
    └── alignment_example.py  # Complete pipeline example
```

**Component Descriptions:**
- **`EntityMatcher`**: LLM-based entity matching logic
- **`CandidateSelector`**: Manages alignment candidates
- **`AttributeExtractor`**: Handles entity attributes
- **`AlignmentPipeline`**: Main orchestration pipeline

### Data Flow

#### 1. Data Loading Phase
```
data/your_dataset/
├── KG1_entity_attributes.json  # KG1 entity attributes
├── KG2_entity_attributes.json  # KG2 entity attributes  
├── alignment_candidates.txt    # Pre-computed candidates
├── ref_pairs                   # Ground truth alignments
└── ent_ids_1, ent_ids_2       # Entity ID mappings
```

#### 2. Entity Matching Phase
For each entity in KG1:
1. **Extract Attributes**: Load entity attributes from JSON
2. **Select Candidates**: Get top-10 candidates from alignment file
3. **LLM Reasoning**: Use LLM to compare entity with candidates
4. **Parse Response**: Extract best match, confidence, and reasoning

#### 3. Evaluation Phase
- Compare predictions against ground truth
- Calculate precision, recall, F1-score
- Generate detailed reports

### Pipeline Steps

#### Step 1: Data Loading
- Load entity attributes from JSON files
- Load alignment candidates from TSV
- Load ground truth pairs
- Initialize entity ID mappings

#### Step 2: Entity Processing
For each entity:
1. Extract entity attributes
2. Get top-10 candidates
3. Format prompt with entity and candidates
4. Generate LLM response
5. Parse response for best match and confidence

#### Step 3: Evaluation
- Compare predictions with ground truth
- Calculate evaluation metrics
- Generate detailed reports

#### Step 4: Output
- Save results to JSON
- Generate evaluation summary
- Log performance metrics

### Key Design Principles

#### 1. **Modularity**
- Each component has a single responsibility
- Clear interfaces between components
- Easy to extend or replace components

#### 2. **Abstraction**
- LLM interface is abstracted for different backends
- Configuration is externalized
- Data loading is separated from processing

#### 3. **Simplicity**
- Focus on core functionality first
- Clear, readable code
- Comprehensive logging and error handling

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

## 🚀 Usage Examples

### Basic Pipeline Usage
```python
from refinEA.pipeline import AlignmentPipeline

# Initialize pipeline
pipeline = AlignmentPipeline(data_dir="data/alignment_files_airelle")

# Align entities
results = pipeline.align_entities(max_entities=10)

# Evaluate results
metrics = pipeline.evaluate_results(results)
print(f"F1 Score: {metrics['f1_score']:.3f}")

# Save results
pipeline.save_results(results, "alignment_results.json")
```

### Single Entity Alignment
```python
from refinEA.matching import EntityMatcher, CandidateSelector, AttributeExtractor

# Initialize components
matcher = EntityMatcher()
selector = CandidateSelector()
extractor = AttributeExtractor()

# Get entity and candidates
entity_attrs = extractor.get_entity_attributes("0", kg_id=1)
candidates = selector.get_candidates("0", max_candidates=10)
candidate_attrs = extractor.get_candidate_attributes([c[0] for c in candidates], kg_id=2)

# Match entity
result = matcher.match_entity("0", entity_attrs, candidate_attrs)
print(f"Best match: {result.best_match_id} (confidence: {result.confidence_score})")
```

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

## 🎯 Key Features

1. **LLM-Based Reasoning**: Uses advanced language models for entity matching
2. **Modular Design**: Easy to extend and modify components
3. **Comprehensive Evaluation**: Multiple metrics and detailed reporting
4. **Error Handling**: Robust error handling and logging
5. **Configuration Management**: Externalized configuration
6. **Batch Processing**: Efficient batch processing of entities

## 🔮 Future Extensions

1. **Multiple LLM Backends**: Support for different LLM providers
2. **Advanced Prompting**: More sophisticated prompt engineering
3. **Ensemble Methods**: Combine multiple matching strategies
4. **Real-time Processing**: Stream processing capabilities
5. **Web Interface**: Web-based UI for alignment
6. **API Endpoints**: REST API for integration

## 📝 Notes

- The system is designed to be **simple** and **modular**
- **LLM reasoning** is the core differentiator
- **Configuration** is externalized for flexibility
- **Error handling** is comprehensive
- **Logging** is detailed for debugging
- **Documentation** is maintained throughout
- All paths are configurable via command-line arguments
- The system is designed for cluster deployment
- Modular architecture for easy extension

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
