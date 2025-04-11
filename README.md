# Quantum vs Classical Computing Benchmark

This research project is to show the comparision between quantum, classical and hybrid computations for machine learning and data anlaysis tasks with detailed performance analytics. A detailed analysis is on our research paper:

##  Key Findings

Our benchmark analysis revealed important insights about different computing paradigms:

### Performance Comparison
- **Accuracy**: Hybrid models consistently outperform both classical and quantum approaches (88-94% vs 85-92% for classical and 75-88% for quantum)
- **Training Speed**: Classical models train fastest (0.5-2.0s), followed by hybrid (2.0-4.0s) and quantum (3.0-6.0s)
- **Resource Usage**: Quantum models require significantly more computational resources than classical models, with hybrid models in between

### Practicality Assessment
- **Setup & Deployment**: Classical models are most practical to implement (7.5/10), followed by hybrid (6.5/10) and quantum (5.0/10)
- **Future Potential**: Quantum and hybrid approaches show promising growth trajectories as hardware and frameworks mature
- **Best Overall Balance**: Hybrid models currently offer the best tradeoff between performance and practicality

## Features
- **Multi-Paradigm Comparison**: Evaluate classical, quantum, and hybrid approaches on equal footing
- **Comprehensive Metrics**: Analyze accuracy, speed, resource usage, and practical implementation factors
- **Beautiful Visualizations**: Apple-inspired UI design for all charts and dashboards
- **Interactive Dashboards**: View results through elegant HTML interfaces
- **Kaggle Integration**: Seamlessly download and use datasets from Kaggle
- **Modular Architecture**: Easily extend with new models, metrics, or visualization types

## Setup and Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/ChahelPaatur/Quantum-vs-classical-computing.git
   cd Quantum-vs-classical-computing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Kaggle API setup** (optional, for downloading datasets)
   - Get your API key from your Kaggle account settings page
   - Create a `kaggle.json` file with your credentials:
     ```json
     {"username":"your_username","key":"your_key"}
     ```
   - Place in `~/.kaggle/` (Unix/MacOS) or `C:\Users\<Windows-username>\.kaggle\` (Windows)
   - Make the file readable only by you: `chmod 600 ~/.kaggle/kaggle.json`

## Quick Start Guide

### Run the Master Dashboard
View the comprehensive visualization dashboard:
```bash
open master_dashboard.html
```

### Run the Ultimate Benchmark
Generate all visualizations and results:
```bash
python ultimate_benchmark_test.py
```

### Run All Benchmarks
Use the unified benchmark runner:
```bash
# Run the default benchmark
python run_all_benchmarks.py

# List all available benchmarks
python run_all_benchmarks.py --list

# Run a specific benchmark
python run_all_benchmarks.py enhanced
```

### Project Cleanup
Organize files and clean up temporary data:
```bash
python cleanup.py
```

## Available Visualizations

The project generates several key visualizations:

1. **Accuracy Comparison**: Bar chart showing prediction accuracy across models
2. **Time Comparison**: Dual-axis chart showing training and inference times
3. **Resource Usage**: Bar chart of memory and CPU requirements
4. **Practicality Metrics**: Radar chart of real-world implementation factors
5. **Multi-Dimensional Analysis**: Radar chart showing all performance dimensions

All visualizations are saved in their respective results directories and accessible through HTML dashboards.

## Project Structure

```
├── data/                          # Datasets directory
│   └── examples/                  # Sample datasets for testing
├── models/                        # Model implementations
│   ├── classical_model.py         # Classical ML models
│   ├── quantum_model.py           # Quantum computing models
│   └── hybrid_model.py            # Combined approaches
├── utils/                         # Utility functions
│   ├── data_loader.py             # Dataset loading utilities
│   ├── metrics.py                 # Performance metrics
│   └── visualization.py           # Visualization tools
├── results/                       # Results from main benchmark
├── visualization_results/         # Basic visualization output
├── fixed_visualizations/          # Enhanced visualization output
├── ultimate_benchmark_results/    # Comprehensive benchmark results
├── benchmark.py                   # Main benchmark script
├── run_all_benchmarks.py          # Unified benchmark runner
├── cleanup.py                     # Project organization script
├── ultimate_benchmark_test.py     # Enhanced benchmark with dashboard
├── master_dashboard.html          # Main visualization dashboard
└── requirements.txt               # Project dependencies
```

## Testing Different Models

### Classical Models
- **Random Forest**: Fast training, good interpretability
- **Gradient Boosting**: Higher accuracy but more complex
- **Neural Network**: Versatile but requires more tuning
- **Deep Learning**: Highest potential accuracy with sufficient data

### Quantum Models
- **Qiskit Variational**: IBM Qiskit-based variational classifier
- **PennyLane Circuit**: PennyLane-based quantum circuits

### Hybrid Models
- **Feature Hybrid**: Split features between quantum and classical
- **Ensemble Hybrid**: Combine predictions from multiple models
- **Quantum Enhanced**: Use quantum computing to enhance classical models

## Interpreting Results

### Dashboard Navigation
The master dashboard organizes results into four sections:
- **Dashboard**: Overview of key metrics with visual summaries
- **Model Comparison**: Detailed breakdown of model performance
- **All Visualizations**: Complete set of visualization outputs
- **Insights**: Key findings and future prospects

### Performance Metrics
- **Accuracy/R²**: Higher is better (prediction performance)
- **Training Time**: Lower is better (model training efficiency)
- **Inference Time**: Lower is better (prediction speed)
- **Memory Usage**: Lower is better (computational efficiency)
- **Practicality Score**: Higher is better (ease of implementation)

## Troubleshooting

### Visualization Issues
If visualizations don't display properly:
1. Run `python specific_visualization_test.py` to generate fixed versions
2. Open `fixed_visualizations/view_fixed.html` to view the corrected charts
3. Use `python cleanup.py` to organize all visualization files

### Memory Problems
For out-of-memory errors:
1. Reduce the number of qubits with `--n_qubits 4` flag
2. Use smaller dataset samples
3. Run only classical models for comparison

### API Connection Issues
For Kaggle API problems:
1. Verify your `kaggle.json` credentials are correct
2. Check your internet connection
3. Use local example datasets from `data/examples/`

## Learn More

- **[WALKTHROUGH.md](WALKTHROUGH.md)**: Detailed step-by-step guide
- **[results/README.md](results/README.md)**: Guide to interpreting benchmark outputs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{QuantumClassicalBenchmark,
  author = {Harshith Chemudugunta, Chahel Paatur},
  title = {Quantum vs Classical Computing Benchmark},
  year = {2025},
  url = {https://github.com/ChahelPaatur/Quantum-vs-classical-computing/},
}
```
