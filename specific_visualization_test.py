"""
This script specifically focuses on fixing and improving the time, practicality, 
and resource usage visualizations that might be problematic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.visualization import AppleStyleVisualizer

def run_specific_visualization_test():
    # Create output directory
    output_dir = "fixed_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = AppleStyleVisualizer()
    visualizer.setup_style()
    
    # Create direct mock data for each visualization type
    
    # ---- TIME COMPARISON DATA ----
    time_results = {
        'Classical - Random Forest': {
            'training_time': 1.2,
            'inference_time': 0.03
        },
        'Classical - Gradient Boosting': {
            'training_time': 1.8, 
            'inference_time': 0.05
        },
        'Quantum - Variational': {
            'training_time': 5.5,
            'inference_time': 0.25
        },
        'Quantum - Circuit': {
            'training_time': 4.2,
            'inference_time': 0.18
        },
        'Hybrid - Feature Based': {
            'training_time': 2.5,
            'inference_time': 0.09
        },
        'Hybrid - Ensemble': {
            'training_time': 3.2,
            'inference_time': 0.12
        }
    }
    
    # ---- RESOURCE USAGE DATA ----
    resource_results = {
        'Classical - Random Forest': {
            'max_memory': 120,
            'avg_memory': 80,
            'max_cpu': 35,
            'avg_cpu': 15,
            'memory_usage': 8,
            'cpu_usage': 15
        },
        'Classical - Gradient Boosting': {
            'max_memory': 150,
            'avg_memory': 90,
            'max_cpu': 40,
            'avg_cpu': 20,
            'memory_usage': 9,
            'cpu_usage': 20
        },
        'Quantum - Variational': {
            'max_memory': 450,
            'avg_memory': 250,
            'max_cpu': 85,
            'avg_cpu': 55,
            'memory_usage': 25,
            'cpu_usage': 55
        },
        'Quantum - Circuit': {
            'max_memory': 420,
            'avg_memory': 230,
            'max_cpu': 75,
            'avg_cpu': 45,
            'memory_usage': 23,
            'cpu_usage': 45
        },
        'Hybrid - Feature Based': {
            'max_memory': 280,
            'avg_memory': 150,
            'max_cpu': 65,
            'avg_cpu': 35,
            'memory_usage': 15,
            'cpu_usage': 35
        },
        'Hybrid - Ensemble': {
            'max_memory': 320,
            'avg_memory': 180,
            'max_cpu': 70,
            'avg_cpu': 40,
            'memory_usage': 18,
            'cpu_usage': 40
        }
    }
    
    # ---- PRACTICALITY DATA ----
    practicality_results = {
        'Classical - Random Forest': {
            'setup_difficulty': 9.0,
            'interpretability': 8.0,
            'deployment_complexity': 8.5,
            'hardware_requirements': 9.0,
            'scalability': 8.0
        },
        'Classical - Gradient Boosting': {
            'setup_difficulty': 8.5,
            'interpretability': 7.5,
            'deployment_complexity': 8.0,
            'hardware_requirements': 8.5,
            'scalability': 7.5
        },
        'Quantum - Variational': {
            'setup_difficulty': 4.0,
            'interpretability': 3.0,
            'deployment_complexity': 3.5,
            'hardware_requirements': 3.0,
            'scalability': 5.0
        },
        'Quantum - Circuit': {
            'setup_difficulty': 3.5,
            'interpretability': 2.5,
            'deployment_complexity': 3.0,
            'hardware_requirements': 2.5,
            'scalability': 4.5
        },
        'Hybrid - Feature Based': {
            'setup_difficulty': 6.0,
            'interpretability': 5.5,
            'deployment_complexity': 5.0,
            'hardware_requirements': 6.0,
            'scalability': 7.0
        },
        'Hybrid - Ensemble': {
            'setup_difficulty': 6.5,
            'interpretability': 6.0,
            'deployment_complexity': 5.5,
            'hardware_requirements': 6.5,
            'scalability': 7.5
        }
    }
    
    # Generate the specific visualizations
    print("Generating specific visualizations...")
    
    # 1. Time comparison
    print("1. Creating time comparison visualization...")
    visualizer.plot_time_comparison(
        time_results,
        save_path=f"{output_dir}/time_comparison.png"
    )
    
    # 2. Resource usage
    print("2. Creating resource usage visualization...")
    visualizer.plot_resource_usage(
        resource_results,
        save_path=f"{output_dir}/resource_usage.png"
    )
    
    # 3. Practicality comparison
    print("3. Creating practicality comparison visualization...")
    visualizer.plot_practicality_comparison(
        practicality_results,
        save_path=f"{output_dir}/practicality_comparison.png"
    )
    
    # Create HTML to view these visualizations
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fixed Visualizations</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f7;
        }}
        h1, h2 {{
            color: #1d1d1f;
        }}
        .container {{
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }}
        img {{
            max-width: 100%;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <h1>Fixed Visualizations</h1>
    
    <div class="container">
        <h2>Training & Inference Time</h2>
        <img src="{output_dir}/time_comparison.png" alt="Time Comparison">
    </div>
    
    <div class="container">
        <h2>Resource Usage</h2>
        <img src="{output_dir}/resource_usage.png" alt="Resource Usage">
    </div>
    
    <div class="container">
        <h2>Practicality Metrics</h2>
        <img src="{output_dir}/practicality_comparison.png" alt="Practicality Comparison">
    </div>
</body>
</html>
    """
    
    # Write the HTML file
    with open(f"{output_dir}/view_fixed.html", "w") as f:
        f.write(html_content)
    
    print(f"\nAll fixed visualizations saved to '{output_dir}' directory!")
    print(f"Open '{output_dir}/view_fixed.html' in your browser to view them.")

if __name__ == "__main__":
    run_specific_visualization_test() 