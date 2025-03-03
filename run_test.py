import argparse
import os
import sys
import pandas as pd
from models.classical_model import get_classical_model_scores
from models.quantum_model import get_quantum_model_scores
from models.hybrid_model import get_hybrid_model_scores
from utils.visualization import AppleStyleVisualizer
from utils.metrics import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run benchmark with local test dataset')
    parser.add_argument('--task', type=str, default='classification', 
                        choices=['classification', 'regression'],
                        help='Type of task (classification or regression)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save results')
    parser.add_argument('--n_qubits', type=int, default=4,
                        help='Number of qubits to use for quantum models')
    parser.add_argument('--test_dataset', type=str, default='data/examples/test_iris.csv',
                        help='Path to test dataset')
    return parser.parse_args()

def load_test_dataset(file_path, task_type='classification'):
    """Load test dataset without using Kaggle API"""
    try:
        # Load the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None, None
        
        # Identify target column (assuming it's the last column)
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode target for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None, None

def run_test_benchmark():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    print(f"Loading test dataset: {args.test_dataset}")
    X, y = load_test_dataset(args.test_dataset, args.task)
    if X is None or y is None:
        sys.exit(1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset loaded. Shape: {X.shape}")
    print(f"Task type: {args.task}")
    
    # Define simplified model configurations
    classical_models = ['random_forest']
    quantum_configs = [{'framework': 'qiskit', 'model_type': 'variational', 'n_qubits': args.n_qubits}]
    hybrid_configs = [{'hybrid_type': 'feature_hybrid', 'n_qubits': args.n_qubits}]
    
    # Run evaluations
    print("Training and evaluating models...")
    classical_results = get_classical_model_scores(classical_models, X_train, X_test, y_train, y_test, args.task)
    quantum_results = get_quantum_model_scores(quantum_configs, X_train, X_test, y_train, y_test, args.task)
    hybrid_results = get_hybrid_model_scores(hybrid_configs, X_train, X_test, y_train, y_test, args.task)
    
    # Combine results
    all_results = {**classical_results, **quantum_results, **hybrid_results}
    
    # Generate visualizations
    visualizer = AppleStyleVisualizer()
    evaluator = ModelEvaluator()
    
    # Save comparative visualizations
    visualizer.setup_style()
    
    if args.task == 'classification':
        visualizer.plot_accuracy_comparison(all_results, 'accuracy', save_path=f"{args.output_dir}/accuracy_comparison.png")
    else:
        visualizer.plot_accuracy_comparison(all_results, 'r2_score', 'R² Score Comparison', save_path=f"{args.output_dir}/r2_comparison.png")
    
    visualizer.plot_time_comparison(all_results, save_path=f"{args.output_dir}/time_comparison.png")
    visualizer.plot_comprehensive_dashboard(all_results, args.task, save_path=f"{args.output_dir}/comprehensive_dashboard.png")
    
    # Save metrics to CSV
    metrics_df = evaluator.get_comparative_metrics()
    metrics_df.to_csv(f"{args.output_dir}/metrics_comparison.csv", index=False)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Test benchmark completed successfully!")

if __name__ == "__main__":
    run_test_benchmark() 