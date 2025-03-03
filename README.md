# Quantum vs Classical Computing Benchmark

This research project compares quantum, classical, and hybrid computing models on various machine learning tasks using Kaggle datasets. The models are evaluated based on accuracy, computational power requirements, execution time, prediction capabilities, forecasting abilities, and practical applicability.

![Benchmark Dashboard](https://raw.githubusercontent.com/example/quantum-vs-classical-computing/main/images/dashboard_example.png)

## Features

- **Comprehensive Comparison**: Compare traditional machine learning algorithms, quantum machine learning approaches, and hybrid models
- **Multiple Metrics**: Evaluate models on accuracy, power consumption, execution time, etc.
- **Beautiful Visualizations**: Apple-inspired UI design for all charts and dashboards
- **Kaggle Integration**: Download and use datasets from Kaggle with built-in API support
- **Modular Design**: Easily extend with new models, metrics, or visualization types

## Project Structure

```
├── data/                     # Directory for storing datasets
├── models/
│   ├── classical_model.py    # Classical computing implementation
│   ├── quantum_model.py      # Quantum computing implementation
│   └── hybrid_model.py       # Hybrid computing implementation
├── utils/
│   ├── data_loader.py        # Utilities for loading Kaggle datasets
│   ├── visualization.py      # Visualization tools for results
│   └── metrics.py            # Performance metrics calculation
├── benchmark.py              # Main benchmarking script
├── run_demo.py               # Simplified demo script
├── results/                  # Directory for storing results
└── requirements.txt          # Dependencies
```

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up Kaggle API credentials (only needed for using Kaggle datasets):
   - Download your Kaggle API key from your Kaggle account settings
   - Place the `kaggle.json` file in `~/.kaggle/` on Unix/MacOS or `C:\Users\<Windows-username>\.kaggle\` on Windows

## Quick Start

Run the demo script to try a simplified benchmark with minimal models:

```bash
python run_demo.py
```

## Running the Full Benchmark

```bash
python benchmark.py --dataset [dataset_name] --task [classification/regression]
```

For more options and detailed usage instructions, see [WALKTHROUGH.md](WALKTHROUGH.md).

## Models Overview

1. **Classical Models**: Implements traditional machine learning algorithms using scikit-learn and TensorFlow.
   - Random Forest
   - Gradient Boosting
   - Neural Networks
   - Support Vector Machines
   - Deep Learning (TensorFlow)

2. **Quantum Models**: Uses quantum computing frameworks (Qiskit, PennyLane) to implement quantum machine learning algorithms.
   - Variational Quantum Classifiers/Regressors
   - Quantum Kernel Methods
   - Quantum Neural Networks

3. **Hybrid Models**: Combines classical and quantum approaches for potentially improved performance.
   - Feature Hybrid (split features between quantum and classical processing)
   - Ensemble Hybrid (combine predictions from multiple models)
   - Quantum-Enhanced (use quantum computing to enhance classical models)

## Evaluation Metrics

- **Accuracy**: Model prediction accuracy
- **Power Efficiency**: Computational resources required
- **Time Efficiency**: Execution time for training and inference
- **Prediction Quality**: Performance on standard prediction tasks
- **Forecasting Ability**: Performance on time-series and forecasting tasks
- **Practicality**: Ease of implementation and deployment in real-world scenarios

## Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn
- TensorFlow
- Qiskit, Qiskit Machine Learning
- PennyLane
- Matplotlib, Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{QuantumClassicalBenchmark,
  author = {Quantum vs Classical Research Team},
  title = {Quantum vs Classical Computing Benchmark},
  year = {2023},
  url = {https://github.com/example/quantum-vs-classical-computing},
}
```

## Running Benchmarks

This project includes several benchmark and visualization options to compare classical, quantum, and hybrid computing approaches:

### Unified Benchmark Runner

The easiest way to run benchmarks is with the unified benchmark runner:

```bash
python run_all_benchmarks.py
```

This will run the comprehensive "ultimate" benchmark by default. For more options:

```bash
# List all available benchmarks
python run_all_benchmarks.py --list

# Run a specific benchmark
python run_all_benchmarks.py enhanced

# Run all benchmarks in sequence
python run_all_benchmarks.py all
```

### Available Benchmark Options

1. **Ultimate Benchmark** - The most comprehensive option with a complete interactive dashboard
   ```bash
   python ultimate_benchmark_test.py
   ```

2. **Enhanced Visualizations** - Improved visualizations with mock data
   ```bash
   python enhanced_visualization_test.py
   ```

3. **Specific Visualizations** - Focused on time, resources, and practicality metrics
   ```bash
   python specific_visualization_test.py
   ```

4. **Standard Visualization** - Basic visualization viewer
   ```bash
   python view_visualizations.py
   ```

## Visualization Outputs

The benchmarks produce various visualization outputs:

- **Accuracy Comparison** - Compare prediction accuracy across models
- **Time Comparison** - Training and inference time for each model
- **Resource Usage** - Memory and CPU requirements
- **Practicality Metrics** - Setup difficulty, interpretability, deployment complexity, etc.
- **Multi-Dimensional Comparison** - Radar chart showing performance across multiple metrics

The visualizations are saved in model-specific directories and can be viewed through HTML dashboards.
