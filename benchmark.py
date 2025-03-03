#!/usr/bin/env python3
"""
Benchmark script for comparing classical, quantum, and hybrid machine learning models.
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Import our modules
from utils.data_loader import DataLoader
from utils.metrics import ModelEvaluator
from utils.visualization import AppleStyleVisualizer
from models.classical_model import get_classical_model_scores
from models.quantum_model import get_quantum_model_scores
from models.hybrid_model import get_hybrid_model_scores


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark classical vs quantum vs hybrid ML models')
    
    parser.add_argument('--dataset', type=str, default=None,
                        help='Kaggle dataset to use (format: "owner/dataset-name") or path to local dataset')
    
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'],
                        help='Type of machine learning task')
    
    parser.add_argument('--classical_models', type=str, nargs='+', 
                        default=['random_forest', 'gradient_boosting', 'neural_network'],
                        help='Classical models to evaluate')
    
    parser.add_argument('--quantum_frameworks', type=str, nargs='+', 
                        default=['qiskit'],
                        help='Quantum frameworks to use (qiskit, pennylane)')
    
    parser.add_argument('--quantum_model_types', type=str, nargs='+', 
                        default=['variational'],
                        help='Quantum model types to evaluate (variational, kernel, circuit)')
    
    parser.add_argument('--hybrid_types', type=str, nargs='+', 
                        default=['feature_hybrid', 'quantum_enhanced'],
                        help='Hybrid model types to evaluate (feature_hybrid, ensemble, quantum_enhanced)')
    
    parser.add_argument('--n_qubits', type=int, default=None,
                        help='Number of qubits to use for quantum computations')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--save_models', action='store_true',
                        help='Save trained models to disk')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during benchmark')
    
    return parser.parse_args()


def run_benchmark(args):
    """Run the benchmark with the given arguments."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Authenticate with Kaggle if using Kaggle dataset
    if args.dataset and '/' in args.dataset:
        data_loader.authenticate_kaggle()
        
        # Get suggestions if no dataset is provided
        if args.dataset is None:
            suggestions = data_loader.get_dataset_suggestions(args.task)
            print("Suggested datasets for {}:".format(args.task))
            for suggestion in suggestions:
                print(f"  {suggestion}")
            return
        
        # Download the dataset
        dataset_path = data_loader.download_dataset(args.dataset)
        if dataset_path is None:
            print(f"Failed to download dataset: {args.dataset}")
            return
        
        # Find data files in the dataset
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.json')):
                    file_path = os.path.join(root, file)
                    print(f"Found data file: {file_path}")
                    
                    try:
                        # Load the dataset
                        X_train, X_test, y_train, y_test, preprocessor = data_loader.load_dataset(
                            file_path, task_type=args.task
                        )
                        break
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue
            else:
                continue
            break
    else:
        # Use local dataset
        file_path = args.dataset
        if file_path and os.path.isfile(file_path):
            try:
                # Load the dataset
                X_train, X_test, y_train, y_test, preprocessor = data_loader.load_dataset(
                    file_path, task_type=args.task
                )
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                return
        else:
            # If no dataset provided, use a simple example dataset
            print("No dataset provided. Using example dataset.")
            if args.task == 'classification':
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer()
                X, y = data.data, data.target
            else:
                from sklearn.datasets import load_boston
                try:
                    data = load_boston()
                    X, y = data.data, data.target
                except:
                    # If Boston dataset is not available, fallback to diabetes
                    from sklearn.datasets import load_diabetes
                    data = load_diabetes()
                    X, y = data.data, data.target
            
            # Split the data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create a basic preprocessor
            from sklearn.preprocessing import StandardScaler
            preprocessor = StandardScaler()
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
    
    # Initialize metrics evaluator
    evaluator = ModelEvaluator()
    
    # Train and evaluate classical models
    print("\n===== Classical Models =====")
    classical_results = get_classical_model_scores(
        args.classical_models, X_train, X_test, y_train, y_test, task_type=args.task
    )
    
    # Train and evaluate quantum models
    print("\n===== Quantum Models =====")
    quantum_configs = []
    for framework in args.quantum_frameworks:
        for model_type in args.quantum_model_types:
            quantum_configs.append({
                'framework': framework,
                'model_type': model_type,
                'n_qubits': args.n_qubits
            })
    
    quantum_results = get_quantum_model_scores(
        quantum_configs, X_train, X_test, y_train, y_test, task_type=args.task
    )
    
    # Train and evaluate hybrid models
    print("\n===== Hybrid Models =====")
    hybrid_configs = []
    for hybrid_type in args.hybrid_types:
        for framework in args.quantum_frameworks:
            hybrid_configs.append({
                'hybrid_type': hybrid_type,
                'classical_model_type': args.classical_models[0],  # Use first classical model
                'quantum_framework': framework,
                'quantum_model_type': args.quantum_model_types[0],  # Use first quantum model type
                'n_qubits': args.n_qubits
            })
    
    hybrid_results = get_hybrid_model_scores(
        hybrid_configs, X_train, X_test, y_train, y_test, task_type=args.task
    )
    
    # Combine all results
    all_results = {**classical_results, **quantum_results, **hybrid_results}
    
    # Evaluate each model with standardized metrics
    for model_name, results in all_results.items():
        if args.task == 'classification':
            metrics = evaluator.evaluate_classification_model(
                model_name=model_name,
                y_true=y_test,
                y_pred=results['y_pred'],
                y_proba=results.get('y_proba'),
                execution_time=results['training_time'],
                power_metrics=(0, results['model_size'])  # No CPU percentage available
            )
        else:
            metrics = evaluator.evaluate_regression_model(
                model_name=model_name,
                y_true=y_test,
                y_pred=results['y_pred'],
                execution_time=results['training_time'],
                power_metrics=(0, results['model_size'])  # No CPU percentage available
            )
        
        # Add inference time
        metrics['inference_time'] = results['inference_time']
        
        # Add practicality scores
        if 'Classical' in model_name:
            evaluator.evaluate_practicality(
                model_name=model_name,
                setup_difficulty=9,       # Classical models are easy to set up
                interpretability=7,       # Depends on the model type
                deployment_complexity=8,  # Easy to deploy
                hardware_requirements=8,  # Low hardware requirements
                scalability=6             # Good for moderate-sized problems
            )
        elif 'Quantum' in model_name:
            evaluator.evaluate_practicality(
                model_name=model_name,
                setup_difficulty=4,       # Quantum models are complex to set up
                interpretability=3,       # Hard to interpret
                deployment_complexity=3,  # Difficult to deploy
                hardware_requirements=2,  # High hardware requirements
                scalability=5             # Currently limited by available quantum resources
            )
        elif 'Hybrid' in model_name:
            evaluator.evaluate_practicality(
                model_name=model_name,
                setup_difficulty=5,       # Moderate complexity
                interpretability=5,       # Moderate interpretability
                deployment_complexity=5,  # Moderate deployment complexity
                hardware_requirements=5,  # Moderate hardware requirements
                scalability=7             # Can scale by leveraging classical components
            )
    
    # Get comparative metrics
    comparative_metrics = evaluator.get_comparative_metrics()
    
    # Save results to CSV
    results_df = pd.DataFrame(comparative_metrics['metrics'])
    results_df.index = comparative_metrics['model_names']
    results_df.to_csv(os.path.join(args.output_dir, 'metrics_comparison.csv'))
    
    # Create visualizations
    visualizer = AppleStyleVisualizer()
    
    if args.task == 'classification':
        # Accuracy comparison
        acc_fig = visualizer.plot_accuracy_comparison(
            evaluator.results, 
            metric='accuracy', 
            title='Model Accuracy Comparison'
        )
        acc_fig.savefig(os.path.join(args.output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Precision-Recall comparison
        metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
    else:
        # R² comparison
        r2_fig = visualizer.plot_accuracy_comparison(
            evaluator.results, 
            metric='r2', 
            title='Model R² Score Comparison'
        )
        r2_fig.savefig(os.path.join(args.output_dir, 'r2_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Error metrics comparison
        metrics_to_plot = ['mse', 'rmse', 'mae', 'r2']
    
    # Time comparison
    time_fig = visualizer.plot_time_comparison(evaluator.results)
    time_fig.savefig(os.path.join(args.output_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Practicality comparison
    pract_fig = visualizer.plot_practicality_comparison(evaluator.results)
    pract_fig.savefig(os.path.join(args.output_dir, 'practicality_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Radar comparison
    radar_metrics = metrics_to_plot + ['execution_time', 'inference_time']
    radar_fig = visualizer.plot_radar_comparison(evaluator.results, metrics=radar_metrics)
    radar_fig.savefig(os.path.join(args.output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Comprehensive dashboard
    dashboard_fig = visualizer.plot_comprehensive_dashboard(evaluator.results, task_type=args.task)
    dashboard_fig.savefig(os.path.join(args.output_dir, 'comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\n===== Benchmark Summary =====")
    print(f"Task type: {args.task}")
    print(f"Models evaluated: {len(evaluator.results)}")
    print(f"Results saved to: {args.output_dir}")
    
    # Print top models by accuracy/r2
    if args.task == 'classification':
        accuracies = {name: result['accuracy'] for name, result in evaluator.results.items()}
        top_model = max(accuracies.items(), key=lambda x: x[1])
        print(f"\nTop model by accuracy: {top_model[0]} ({top_model[1]:.4f})")
    else:
        r2_scores = {name: result['r2'] for name, result in evaluator.results.items()}
        top_model = max(r2_scores.items(), key=lambda x: x[1])
        print(f"\nTop model by R²: {top_model[0]} ({top_model[1]:.4f})")
    
    # Print top models by execution time
    exec_times = {name: result['execution_time'] for name, result in evaluator.results.items()}
    fastest_model = min(exec_times.items(), key=lambda x: x[1])
    print(f"Fastest model: {fastest_model[0]} ({fastest_model[1]:.2f} seconds)")
    
    # Print top models by inference time
    inf_times = {name: result['inference_time'] for name, result in evaluator.results.items()}
    fastest_inference = min(inf_times.items(), key=lambda x: x[1])
    print(f"Fastest inference: {fastest_inference[0]} ({fastest_inference[1]:.2f} ms per sample)")
    
    # Print top models by practicality
    practicalities = {name: result['practicality']['overall_practicality'] 
                     for name, result in evaluator.results.items() 
                     if 'practicality' in result}
    most_practical = max(practicalities.items(), key=lambda x: x[1])
    print(f"Most practical model: {most_practical[0]} (score: {most_practical[1]:.2f}/10)")


if __name__ == "__main__":
    print("Quantum vs Classical Computing Benchmark")
    print("=======================================")
    
    args = parse_arguments()
    run_benchmark(args) 