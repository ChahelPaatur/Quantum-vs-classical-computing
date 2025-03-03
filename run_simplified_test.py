import argparse
import os
import sys
import pandas as pd
from models.classical_model import get_classical_model_scores
from utils.visualization import AppleStyleVisualizer
from utils.metrics import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run simplified benchmark with local test dataset')
    parser.add_argument('--task', type=str, default='classification', 
                        choices=['classification', 'regression'],
                        help='Type of task (classification or regression)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save results')
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
        
        # Show dataframe info
        print("\nDataset preview:")
        print(df.head())
        print("\nDataset info:")
        print(f"- Rows: {df.shape[0]}")
        print(f"- Columns: {df.shape[1]}")
        print(f"- Column names: {df.columns.tolist()}")
        
        # Identify target column (assuming it's the last column)
        target_col = df.columns[-1]
        print(f"- Using '{target_col}' as target column")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode target for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"- Classes: {le.classes_.tolist()}")
        
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None, None

def run_simplified_test():
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
    
    print(f"\nDataset split complete:")
    print(f"- Training set: {X_train.shape[0]} samples")
    print(f"- Test set: {X_test.shape[0]} samples")
    print(f"- Task type: {args.task}")
    
    # Define simplified model configurations (classical only)
    classical_models = ['random_forest', 'gradient_boosting']
    
    # Run evaluations
    print("\nTraining and evaluating classical models...")
    classical_results = get_classical_model_scores(classical_models, X_train, X_test, y_train, y_test, args.task)
    
    # Generate visualizations
    visualizer = AppleStyleVisualizer()
    evaluator = ModelEvaluator()
    
    # Save comparative visualizations
    visualizer.setup_style()
    
    if args.task == 'classification':
        visualizer.plot_accuracy_comparison(classical_results, 'accuracy', save_path=f"{args.output_dir}/accuracy_comparison.png")
    else:
        visualizer.plot_accuracy_comparison(classical_results, 'r2_score', 'R² Score Comparison', save_path=f"{args.output_dir}/r2_comparison.png")
    
    visualizer.plot_time_comparison(classical_results, save_path=f"{args.output_dir}/time_comparison.png")
    
    # Save metrics to CSV
    metrics = evaluator.get_comparative_metrics()
    if isinstance(metrics, dict):
        # Convert dict to DataFrame
        metrics_df = pd.DataFrame(metrics)
    else:
        metrics_df = metrics
    
    metrics_df.to_csv(f"{args.output_dir}/metrics_comparison.csv", index=False)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Simplified test completed successfully!")

if __name__ == "__main__":
    run_simplified_test() 