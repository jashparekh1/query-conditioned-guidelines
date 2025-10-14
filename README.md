# Query Conditioned Guidelines

A research project exploring query-conditioned guidelines for improving language model performance across various reasoning tasks.

## Overview

This repository contains research code and experiments for developing and evaluating query-conditioned guidelines that help language models perform better on reasoning tasks. We use the VERL framework for training and evaluation.

## Project Structure

```
├── query_conditioned_guidelines/    # Main research code
│   ├── data/                        # Dataset processing and storage
│   │   ├── commonsenseqa/          # CommonsenseQA dataset
│   │   ├── gsm8k/                  # GSM8K math reasoning dataset
│   │   ├── math/                   # MATH dataset
│   │   └── strategyqa/             # StrategyQA dataset
│   ├── models/                      # Model checkpoints and configurations
│   │   ├── checkpoints/            # Training checkpoints
│   │   ├── frozen/                 # Frozen model weights
│   │   └── guideline/              # Guideline-specific models
│   ├── results/                     # Experimental results
│   │   ├── baseline/               # Baseline model results
│   │   └── training/               # Training experiment results
│   ├── scripts/                     # Utility scripts
│   │   ├── baseline_eval/           # Baseline evaluation scripts
│   │   └── download_data/          # Data download utilities
│   ├── src/                         # Source code
│   ├── tests/                       # Test suite
│   └── requirements.txt             # Python dependencies
└── verl/                           # VERL framework (submodule)
    └── ...                         # VERL source code
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <repo-name>
   git submodule update --init --recursive
   ```

2. **Install Python dependencies:**
   ```bash
   cd query_conditioned_guidelines
   pip install -r requirements.txt
   ```

3. **Install VERL framework:**
   ```bash
   cd verl
   pip install -e .
   ```

4. **Download datasets:**
   ```bash
   cd query_conditioned_guidelines
   python scripts/download_data/download_all.py
   ```

## Quick Start

### Running Baseline Evaluations

```bash
cd query_conditioned_guidelines/tests
bash run_gsm8k_baseline.sh
```

### Testing Model Inference

```bash
cd query_conditioned_guidelines/tests
python test_qwen_inference.py
```

## Datasets

This project works with several reasoning datasets:

- **CommonsenseQA**: Commonsense reasoning questions
- **GSM8K**: Grade school math word problems
- **MATH**: Advanced mathematics problems
- **StrategyQA**: Strategic reasoning questions

## Models

The project supports various language models including:
- Qwen models
- Other transformer-based models via the VERL framework

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/`
6. Submit a pull request

## Research

This project is part of ongoing research into query-conditioned guidelines for language models. For more details about the methodology and results, please refer to our research papers (to be added).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the [VERL](https://github.com/volcengine/verl) framework
- Uses datasets from various research communities
- Thanks to all contributors and the open-source community

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your-paper-2024,
  title={Query Conditioned Guidelines for Language Models},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Contact

For questions or collaboration, please contact [your-email@domain.com] or open an issue on GitHub.

## Roadmap

- [ ] Add more reasoning datasets
- [ ] Implement additional guideline strategies
- [ ] Improve evaluation metrics
- [ ] Add visualization tools
- [ ] Create interactive demos
