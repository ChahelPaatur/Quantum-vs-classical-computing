# Quantum vs Classical Computing Benchmark Walkthrough

This project provides a comprehensive framework for comparing quantum, classical, and hybrid machine learning models across various metrics including accuracy, computational power requirements, execution time, prediction capabilities, and practicality.

## Project Overview

The benchmark:
- Trains and evaluates multiple models from each computing paradigm
- Tests models on Kaggle datasets or built-in datasets
- Measures performance across multiple dimensions
- Produces beautiful Apple-inspired visualizations
- Outputs comprehensive comparison reports

## Setup

1. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Kaggle API Setup** (optional, only needed for Kaggle datasets)

   - Create a Kaggle account if you don't have one
   - Generate an API key from your Kaggle account settings
   - Place the `kaggle.json` file in:
     - `~/.kaggle/` on Unix/MacOS
     - `C:\Users\<Windows-username>\.kaggle\` on Windows

## Running the Benchmark

### Quick Demo

For a quick demonstration with minimal models, run:

```bash
python run_demo.py
```

Options:
- `--task classification` or `--task regression`
- `--output_dir path/to/save/results`

Example:
```bash
python run_demo.py --task classification --output_dir my_demo_results
```

### Full Benchmark

For a complete benchmark with all models, run:

```bash
python benchmark.py
```

Key options:
- `--dataset kaggle_owner/dataset_name` or path to a local dataset file
- `--task classification` or `--task regression`
- `--classical_models random_forest gradient_boosting neural_network deep_learning svm`
- `--quantum_frameworks qiskit pennylane`
- `--quantum_model_types variational kernel circuit`
- `--hybrid_types feature_hybrid ensemble quantum_enhanced`
- `--n_qubits 4` (adjust based on your computational resources)
- `--output_dir path/to/save/results`

Example:
```bash
python benchmark.py --dataset uciml/iris --task classification --n_qubits 4
```

## Understanding the Results

After running the benchmark, you'll find the following in your output directory:

1. **CSV Reports**:
   - `metrics_comparison.csv`: Detailed comparison of all metrics across models

2. **Visualizations**:
   - Accuracy/R² comparisons
   - Execution time comparisons
   - Resource usage comparisons
   - Practicality metric comparisons
   - Radar charts showing multiple metrics together
   - A comprehensive dashboard

3. **Terminal Summary**:
   - Top-performing models by different metrics
   - Overall benchmark statistics

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

## Customizing the Benchmark

The benchmark is designed to be highly modular. You can extend it by:

1. **Adding new models**:
   - Extend the `ClassicalModel`, `QuantumModel`, or `HybridModel` classes

2. **Adding new datasets**:
   - Place your own datasets in the `data/` directory or use Kaggle datasets

3. **Adding new metrics**:
   - Extend the `ModelEvaluator` class in `utils/metrics.py`

4. **Creating new visualizations**:
   - Add new plotting functions to `AppleStyleVisualizer` in `utils/visualization.py`

## Troubleshooting

- **Memory issues**: Reduce the number of qubits with `--n_qubits`
- **Kaggle API errors**: Ensure your `kaggle.json` is placed in the correct directory
- **Quantum framework errors**: Check that you have the correct versions of Qiskit and PennyLane

## Learning More

To learn more about the computing paradigms compared in this benchmark:

- **Classical computing**: Refers to traditional algorithms running on conventional hardware
- **Quantum computing**: Uses quantum mechanical phenomena to perform computations
- **Hybrid computing**: Combines classical and quantum approaches

Each has its strengths and limitations, which this benchmark helps to quantify and visualize. 