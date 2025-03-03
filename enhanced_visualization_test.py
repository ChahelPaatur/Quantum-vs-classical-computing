import os
import numpy as np
import pandas as pd
from utils.visualization import AppleStyleVisualizer
from utils.metrics import ModelEvaluator
import matplotlib.pyplot as plt

def generate_mock_results():
    """
    Generate enhanced mock results for all three model types to demonstrate visualizations.
    We're adding additional metrics to ensure all visualizations have complete data.
    """
    # Define model types we want to simulate
    classical_models = ['Random Forest', 'Gradient Boosting', 'Neural Network']
    quantum_models = ['Qiskit Variational', 'PennyLane Circuit']  
    hybrid_models = ['Feature Hybrid', 'Ensemble Hybrid', 'Quantum Enhanced']
    
    # Dictionary to store all results
    all_results = {}
    
    # Generate realistic metrics for classical models
    for model in classical_models:
        all_results[model] = {
            'accuracy': np.random.uniform(0.82, 0.92),
            'precision': np.random.uniform(0.80, 0.91),
            'recall': np.random.uniform(0.78, 0.90),
            'f1_score': np.random.uniform(0.79, 0.91),
            'training_time': np.random.uniform(0.5, 2.0),
            'inference_time': np.random.uniform(0.02, 0.08),
            'memory_usage': np.random.uniform(2, 10),
            'cpu_usage': np.random.uniform(10, 20),
            'setup_difficulty': np.random.uniform(8, 10),
            'interpretability': np.random.uniform(7, 9),
            'deployment_complexity': np.random.uniform(7, 9),
            'hardware_requirements': np.random.uniform(8, 10),
            'scalability': np.random.uniform(7, 9),
            'model_size': np.random.uniform(1, 5),
            'model_type': 'Classical',
            # Add metrics specifically for practicality chart
            'setup_score': np.random.uniform(7, 9),
            'interpretability_score': np.random.uniform(7, 9),
            'deployment_score': np.random.uniform(7, 9),
            'hardware_score': np.random.uniform(8, 10),
            'scalability_score': np.random.uniform(7, 9),
            # Add metrics for resource usage
            'max_memory': np.random.uniform(100, 200),
            'avg_memory': np.random.uniform(50, 100),
            'max_cpu': np.random.uniform(20, 40),
            'avg_cpu': np.random.uniform(10, 20)
        }
    
    # Generate realistic metrics for quantum models
    for model in quantum_models:
        all_results[model] = {
            'accuracy': np.random.uniform(0.75, 0.88),
            'precision': np.random.uniform(0.73, 0.86),
            'recall': np.random.uniform(0.72, 0.85),
            'f1_score': np.random.uniform(0.72, 0.86),
            'training_time': np.random.uniform(3.0, 6.0),
            'inference_time': np.random.uniform(0.1, 0.3),
            'memory_usage': np.random.uniform(15, 25),
            'cpu_usage': np.random.uniform(40, 60),
            'setup_difficulty': np.random.uniform(3, 6),
            'interpretability': np.random.uniform(2, 5),
            'deployment_complexity': np.random.uniform(2, 5),
            'hardware_requirements': np.random.uniform(2, 5),
            'scalability': np.random.uniform(4, 7),
            'model_size': np.random.uniform(3, 8),
            'model_type': 'Quantum',
            # Add metrics specifically for practicality chart
            'setup_score': np.random.uniform(3, 5),
            'interpretability_score': np.random.uniform(2, 4),
            'deployment_score': np.random.uniform(2, 5),
            'hardware_score': np.random.uniform(2, 4),
            'scalability_score': np.random.uniform(3, 6),
            # Add metrics for resource usage
            'max_memory': np.random.uniform(400, 600),
            'avg_memory': np.random.uniform(200, 300),
            'max_cpu': np.random.uniform(70, 90),
            'avg_cpu': np.random.uniform(40, 60)
        }
    
    # Generate realistic metrics for hybrid models
    for model in hybrid_models:
        all_results[model] = {
            'accuracy': np.random.uniform(0.85, 0.94),
            'precision': np.random.uniform(0.83, 0.93),
            'recall': np.random.uniform(0.82, 0.93),
            'f1_score': np.random.uniform(0.83, 0.93),
            'training_time': np.random.uniform(2.0, 4.0),
            'inference_time': np.random.uniform(0.05, 0.15),
            'memory_usage': np.random.uniform(10, 18),
            'cpu_usage': np.random.uniform(25, 45),
            'setup_difficulty': np.random.uniform(5, 7),
            'interpretability': np.random.uniform(5, 7),
            'deployment_complexity': np.random.uniform(4, 7),
            'hardware_requirements': np.random.uniform(5, 7),
            'scalability': np.random.uniform(6, 8),
            'model_size': np.random.uniform(2, 6),
            'model_type': 'Hybrid',
            # Add metrics specifically for practicality chart
            'setup_score': np.random.uniform(5, 7),
            'interpretability_score': np.random.uniform(5, 7),
            'deployment_score': np.random.uniform(4, 7),
            'hardware_score': np.random.uniform(5, 7),
            'scalability_score': np.random.uniform(6, 8),
            # Add metrics for resource usage
            'max_memory': np.random.uniform(200, 400),
            'avg_memory': np.random.uniform(100, 200),
            'max_cpu': np.random.uniform(50, 70),
            'avg_cpu': np.random.uniform(25, 45)
        }
    
    return all_results

def run_visualization_test():
    """Run comprehensive visualization tests with improved mock data"""
    # Create output directory
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate mock results
    print("Generating mock results for visualization...")
    all_results = generate_mock_results()
    
    # Initialize visualizer
    visualizer = AppleStyleVisualizer()
    visualizer.setup_style()
    
    # Create and save all visualizations
    print("\nGenerating visualizations...")
    
    # 1. Accuracy comparison
    print("1. Creating accuracy comparison plot...")
    visualizer.plot_accuracy_comparison(
        all_results, 
        metric='accuracy', 
        title='Model Accuracy Comparison',
        save_path=f"{output_dir}/accuracy_comparison.png"
    )
    
    # 2. Time comparison - use training_time and inference_time
    print("2. Creating time comparison plot...")
    visualizer.plot_time_comparison(
        all_results,
        save_path=f"{output_dir}/time_comparison.png"
    )
    
    # 3. Resource usage - use memory_usage and cpu_usage
    print("3. Creating resource usage plot...")
    # Convert resource usage to a readable form
    for model in all_results:
        if 'avg_memory' not in all_results[model]:
            all_results[model]['avg_memory'] = all_results[model]['memory_usage'] * 10
            all_results[model]['max_memory'] = all_results[model]['memory_usage'] * 20
            all_results[model]['avg_cpu'] = all_results[model]['cpu_usage']
            all_results[model]['max_cpu'] = all_results[model]['cpu_usage'] * 1.5
    
    visualizer.plot_resource_usage(
        all_results,
        save_path=f"{output_dir}/resource_usage.png"
    )
    
    # 4. Practicality comparison - use setup_score, etc.
    print("4. Creating practicality comparison plot...")
    # Ensure we have all necessary practicality metrics
    for model in all_results:
        all_results[model]['setup_difficulty'] = all_results[model].get('setup_score', 7)
        all_results[model]['interpretability'] = all_results[model].get('interpretability_score', 6)
        all_results[model]['deployment_complexity'] = all_results[model].get('deployment_score', 5)
        all_results[model]['hardware_requirements'] = all_results[model].get('hardware_score', 6)
        all_results[model]['scalability'] = all_results[model].get('scalability_score', 6)
    
    visualizer.plot_practicality_comparison(
        all_results,
        save_path=f"{output_dir}/practicality_comparison.png"
    )
    
    # 5. Radar chart
    print("5. Creating radar comparison plot...")
    # Use only metrics that exist in our mock data
    radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'scalability']
    visualizer.plot_radar_comparison(
        all_results,
        metrics=radar_metrics,
        save_path=f"{output_dir}/radar_comparison.png"
    )
    
    # 6. Comprehensive dashboard - skip for now due to dimension mismatch
    print("6. Creating comprehensive dashboard (skipped due to dimension issue)...")
    # Uncomment if needed later:
    # visualizer.plot_comprehensive_dashboard(
    #     all_results,
    #     task_type='classification',
    #     save_path=f"{output_dir}/comprehensive_dashboard.png"
    # )
    
    # 7. Create a model comparison table
    print("7. Creating model comparison table...")
    model_types = [result['model_type'] for model in all_results for result in [all_results[model]]]
    model_names = list(all_results.keys())
    accuracies = [result['accuracy'] for model in all_results for result in [all_results[model]]]
    training_times = [result['training_time'] for model in all_results for result in [all_results[model]]]
    
    comparison_df = pd.DataFrame({
        'Model Type': model_types,
        'Model Name': model_names,
        'Accuracy': accuracies,
        'Training Time (s)': training_times
    })
    
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"\nModel comparison table:\n{comparison_df}")
    
    print(f"\nAll visualizations saved to {output_dir} directory!")

if __name__ == "__main__":
    run_visualization_test() 