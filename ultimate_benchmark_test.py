"""
Ultimate Benchmark Test - Quantum vs Classical vs Hybrid Computing

This script runs a comprehensive visualization benchmark showing the comparison
between classical, quantum and hybrid computing approaches across multiple dimensions.
It creates a complete set of visualizations and an interactive dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.visualization import AppleStyleVisualizer
from utils.metrics import ModelEvaluator
import webbrowser
import json
from datetime import datetime

# Configuration
OUTPUT_DIR = "ultimate_benchmark_results"
MODELS_CONFIG = {
    "Classical": {
        "models": ["Random Forest", "Gradient Boosting", "Neural Network", "Deep Learning"],
        "color": "#5470c6",  # Blue
        "accuracy_range": (0.82, 0.92),
        "training_time_range": (0.5, 2.0),
        "inference_time_range": (0.02, 0.08),
        "memory_range": (2, 10),
        "cpu_range": (10, 20),
        "practicality_range": (7, 9),
        "description": "Traditional algorithms running on conventional hardware"
    },
    "Quantum": {
        "models": ["Qiskit Variational", "PennyLane Circuit", "Quantum Kernel", "Quantum Neural Net"],
        "color": "#91cc75",  # Green
        "accuracy_range": (0.75, 0.88),
        "training_time_range": (3.0, 6.0),
        "inference_time_range": (0.1, 0.3),
        "memory_range": (15, 25),
        "cpu_range": (40, 60),
        "practicality_range": (3, 6),
        "description": "Algorithms that leverage quantum mechanical principles for computation"
    },
    "Hybrid": {
        "models": ["Feature Hybrid", "Ensemble Hybrid", "Quantum Enhanced", "Quantum-Classical"],
        "color": "#fac858",  # Yellow
        "accuracy_range": (0.85, 0.94),
        "training_time_range": (2.0, 4.0),
        "inference_time_range": (0.05, 0.15),
        "memory_range": (10, 18),
        "cpu_range": (25, 45),
        "practicality_range": (5, 7),
        "description": "Combines classical and quantum approaches to maximize strengths of both"
    }
}

def generate_mock_results():
    """
    Generate comprehensive mock results for all three model types.
    """
    all_results = {}
    
    for paradigm, config in MODELS_CONFIG.items():
        for model in config["models"]:
            model_name = f"{model}"
            
            # Basic metrics
            accuracy = np.random.uniform(*config["accuracy_range"])
            training_time = np.random.uniform(*config["training_time_range"])
            inference_time = np.random.uniform(*config["inference_time_range"])
            memory_usage = np.random.uniform(*config["memory_range"])
            cpu_usage = np.random.uniform(*config["cpu_range"])
            
            # Classification metrics
            precision = np.random.uniform(accuracy - 0.05, accuracy + 0.03)
            recall = np.random.uniform(accuracy - 0.06, accuracy + 0.02)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            # Resource metrics
            max_memory = memory_usage * np.random.uniform(1.8, 2.2)
            avg_memory = memory_usage * np.random.uniform(0.8, 1.2)
            max_cpu = cpu_usage * np.random.uniform(1.4, 1.8)
            avg_cpu = cpu_usage * np.random.uniform(0.9, 1.1)
            
            # Practicality metrics 
            prac_base = np.random.uniform(*config["practicality_range"])
            setup_difficulty = prac_base + np.random.uniform(-1.0, 1.0)
            interpretability = prac_base + np.random.uniform(-1.0, 1.0)
            deployment_complexity = prac_base + np.random.uniform(-0.8, 0.8)
            hardware_requirements = prac_base + np.random.uniform(-0.5, 1.2)
            scalability = prac_base + np.random.uniform(-0.7, 0.7)
            
            # Ensure values are within reasonable ranges
            setup_difficulty = max(min(setup_difficulty, 10.0), 1.0)
            interpretability = max(min(interpretability, 10.0), 1.0)
            deployment_complexity = max(min(deployment_complexity, 10.0), 1.0)
            hardware_requirements = max(min(hardware_requirements, 10.0), 1.0)
            scalability = max(min(scalability, 10.0), 1.0)
            
            all_results[model_name] = {
                'model_type': paradigm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'training_time': training_time,
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'max_memory': max_memory,
                'avg_memory': avg_memory,
                'max_cpu': max_cpu,
                'avg_cpu': avg_cpu,
                'setup_difficulty': setup_difficulty,
                'interpretability': interpretability,
                'deployment_complexity': deployment_complexity,
                'hardware_requirements': hardware_requirements,
                'scalability': scalability,
                'model_size': np.random.uniform(1, 5) if paradigm == "Classical" else 
                              np.random.uniform(3, 8) if paradigm == "Quantum" else
                              np.random.uniform(2, 6)
            }
    
    return all_results

def create_benchmark_summary(results, output_dir):
    """Create a summary of benchmark results"""
    # Group results by model type
    model_types = {}
    for model, metrics in results.items():
        model_type = metrics['model_type']
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append((model, metrics))
    
    # Calculate averages for each model type
    averages = {}
    for model_type, models in model_types.items():
        type_metrics = {
            'accuracy': np.mean([m[1]['accuracy'] for m in models]),
            'training_time': np.mean([m[1]['training_time'] for m in models]),
            'inference_time': np.mean([m[1]['inference_time'] for m in models]),
            'memory_usage': np.mean([m[1]['memory_usage'] for m in models]),
            'cpu_usage': np.mean([m[1]['cpu_usage'] for m in models]),
            'setup_difficulty': np.mean([m[1]['setup_difficulty'] for m in models]),
            'interpretability': np.mean([m[1]['interpretability'] for m in models]),
            'deployment_complexity': np.mean([m[1]['deployment_complexity'] for m in models]),
            'hardware_requirements': np.mean([m[1]['hardware_requirements'] for m in models]),
            'scalability': np.mean([m[1]['scalability'] for m in models]),
        }
        averages[model_type] = type_metrics
    
    # Find best model for each metric
    best_models = {
        'accuracy': max(results.items(), key=lambda x: x[1]['accuracy']),
        'training_time': min(results.items(), key=lambda x: x[1]['training_time']),
        'inference_time': min(results.items(), key=lambda x: x[1]['inference_time']),
        'memory_usage': min(results.items(), key=lambda x: x[1]['memory_usage']),
        'practicality_score': max(results.items(), key=lambda x: sum([
            x[1]['setup_difficulty'], 
            x[1]['interpretability'],
            x[1]['deployment_complexity'],
            x[1]['hardware_requirements'],
            x[1]['scalability']
        ]))
    }
    
    # Generate insights
    insights = {
        'Classical': "Classical models provide good accuracy with fast training times and are easier to deploy and interpret.",
        'Quantum': "Quantum models show promise but currently have higher resource requirements and complexity.",
        'Hybrid': "Hybrid models often achieve the best accuracy by combining the strengths of both classical and quantum approaches."
    }
    
    # Save summary as JSON
    summary = {
        'averages': averages,
        'best_models': {k: {'model': v[0], 'value': v[1][k if k != 'practicality_score' else 'setup_difficulty']} for k, v in best_models.items()},
        'insights': insights,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{output_dir}/benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def run_ultimate_benchmark():
    """Run the ultimate benchmark test with all visualizations"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📊 Running ultimate benchmark test...")
    print(f"🔍 Results will be saved to: {OUTPUT_DIR}\n")
    
    # Generate mock results
    print("🧪 Generating model results...")
    all_results = generate_mock_results()
    
    # Save raw data
    results_df = pd.DataFrame([
        {**{'model': model}, **metrics}
        for model, metrics in all_results.items()
    ])
    results_df.to_csv(f"{OUTPUT_DIR}/all_results.csv", index=False)
    
    # Initialize visualizer
    visualizer = AppleStyleVisualizer()
    visualizer.setup_style()
    
    print("\n🎨 Creating visualizations...")
    
    # Create all visualizations
    visualizations = [
        # 1. Accuracy comparison
        ("accuracy_comparison.png", "Model Accuracy", lambda: visualizer.plot_accuracy_comparison(
            all_results, 'accuracy', 'Model Accuracy Comparison',
            save_path=f"{OUTPUT_DIR}/accuracy_comparison.png"
        )),
        
        # 2. Time comparison
        ("time_comparison.png", "Time Efficiency", lambda: visualizer.plot_time_comparison(
            all_results, save_path=f"{OUTPUT_DIR}/time_comparison.png"
        )),
        
        # 3. Resource usage
        ("resource_usage.png", "Resource Usage", lambda: visualizer.plot_resource_usage(
            all_results, save_path=f"{OUTPUT_DIR}/resource_usage.png"
        )),
        
        # 4. Practicality comparison
        ("practicality_comparison.png", "Practicality Metrics", lambda: visualizer.plot_practicality_comparison(
            all_results, save_path=f"{OUTPUT_DIR}/practicality_comparison.png"
        )),
        
        # 5. Radar comparison
        ("radar_comparison.png", "Multi-Dimensional View", lambda: visualizer.plot_radar_comparison(
            all_results, 
            metrics=['accuracy', 'f1_score', 'scalability', 'interpretability', 'setup_difficulty'],
            save_path=f"{OUTPUT_DIR}/radar_comparison.png"
        ))
    ]
    
    # Execute each visualization
    for filename, title, viz_func in visualizations:
        print(f"  • Creating {title} visualization...")
        viz_func()
    
    # Create benchmark summary
    print("\n📋 Generating benchmark summary...")
    summary = create_benchmark_summary(all_results, OUTPUT_DIR)
    
    # Generate detailed model comparison table
    print("📑 Creating model comparison table...")
    comparison_columns = [
        'model',
        'model_type',
        'accuracy',
        'training_time',
        'inference_time',
        'memory_usage',
        'cpu_usage',
        'setup_difficulty',
        'interpretability',
        'deployment_complexity',
        'hardware_requirements',
        'scalability'
    ]
    comparison_df = results_df[comparison_columns]
    comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    
    # Create HTML dashboard
    print("🖥️ Generating interactive dashboard...")
    
    # Get best model for each category
    best_accuracy = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    best_training = min(all_results.items(), key=lambda x: x[1]['training_time'])
    best_inference = min(all_results.items(), key=lambda x: x[1]['inference_time'])
    best_memory = min(all_results.items(), key=lambda x: x[1]['memory_usage'])
    best_practicality = max(all_results.items(), key=lambda x: sum([
        x[1]['setup_difficulty'], 
        x[1]['interpretability'],
        x[1]['deployment_complexity'],
        x[1]['hardware_requirements'],
        x[1]['scalability']
    ]) / 5)
    
    # Prepare data for dashboard
    model_rows = ""
    for model, metrics in all_results.items():
        model_type = metrics['model_type']
        accuracy = f"{metrics['accuracy']:.2%}"
        training_time = f"{metrics['training_time']:.3f}s"
        inference_time = f"{metrics['inference_time'] * 1000:.2f}ms"
        memory = f"{metrics['memory_usage']:.1f}MB"
        
        # Determine if this model is a "best" in any category
        badges = []
        if (model, metrics) == best_accuracy:
            badges.append("Most Accurate")
        if (model, metrics) == best_training:
            badges.append("Fastest Training")
        if (model, metrics) == best_inference:
            badges.append("Fastest Inference")
        if (model, metrics) == best_memory:
            badges.append("Most Efficient")
        if (model, metrics) == best_practicality:
            badges.append("Most Practical")
        
        badge_html = ""
        if badges:
            badge_html = ''.join([f'<span class="badge">{badge}</span>' for badge in badges])
        
        model_rows += f"""
            <tr>
                <td>
                    <div class="model-name">{model}</div>
                    <div class="model-type" style="color: {MODELS_CONFIG[model_type]['color']}">{model_type}</div>
                </td>
                <td>{accuracy}</td>
                <td>{training_time}</td>
                <td>{inference_time}</td>
                <td>{memory}</td>
                <td>{badge_html}</td>
            </tr>
        """
    
    # Get averages for type comparison
    type_comparison = ""
    for model_type, metrics in summary['averages'].items():
        accuracy = f"{metrics['accuracy']:.2%}"
        training_time = f"{metrics['training_time']:.3f}s"
        inference_time = f"{metrics['inference_time'] * 1000:.2f}ms"
        memory = f"{metrics['memory_usage']:.1f}MB"
        description = MODELS_CONFIG[model_type]['description']
        
        type_comparison += f"""
            <div class="paradigm-card" style="border-color: {MODELS_CONFIG[model_type]['color']}">
                <h3>{model_type} Computing</h3>
                <p class="description">{description}</p>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">{accuracy}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{training_time}</div>
                        <div class="metric-label">Training</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{inference_time}</div>
                        <div class="metric-label">Inference</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{memory}</div>
                        <div class="metric-label">Memory</div>
                    </div>
                </div>
                <div class="insight">{summary['insights'][model_type]}</div>
            </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum vs Classical vs Hybrid Computing Benchmark</title>
    <style>
        /* Apple-inspired styling */
        :root {{
            --apple-blue: #0071e3;
            --apple-gray: #f5f5f7;
            --apple-dark: #1d1d1f;
            --apple-green: #68cc45;
            --classical-color: #5470c6;
            --quantum-color: #91cc75;
            --hybrid-color: #fac858;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: var(--apple-dark);
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--apple-gray);
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        h3 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        p.subtitle {{
            font-size: 1.2rem;
            color: #515154;
            margin-bottom: 0;
        }}
        
        .container {{
            background-color: white;
            border-radius: 18px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .visualizations-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .visualization {{
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            overflow: hidden;
            transition: transform 0.3s ease;
        }}
        
        .visualization:hover {{
            transform: translateY(-5px);
        }}
        
        .visualization-title {{
            padding: 15px 20px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .visualization-title h3 {{
            margin: 0;
            font-size: 1.2rem;
        }}
        
        .visualization img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .models-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .models-table th {{
            background-color: #f8f8f8;
            text-align: left;
            padding: 12px 15px;
            font-weight: 600;
        }}
        
        .models-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .model-name {{
            font-weight: 600;
            margin-bottom: 3px;
        }}
        
        .model-type {{
            font-size: 0.85rem;
            opacity: 0.8;
        }}
        
        .badge {{
            display: inline-block;
            background-color: var(--apple-blue);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        .paradigms-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}
        
        .paradigm-card {{
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            padding: 1.5rem;
            border-top: 4px solid;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        
        .metric {{
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 1.3rem;
            font-weight: 600;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: #515154;
        }}
        
        .description {{
            color: #515154;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }}
        
        .insight {{
            background-color: #f8f8f8;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.4;
        }}
        
        footer {{
            text-align: center;
            margin-top: 3rem;
            color: #86868b;
            font-size: 0.9rem;
            padding-top: 1rem;
            border-top: 1px solid #eaeaea;
        }}
        
        .icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background-color: var(--apple-blue);
            color: white;
        }}
        
        .tab-container {{
            margin-bottom: 2rem;
        }}
        
        .tab-buttons {{
            display: flex;
            margin-bottom: 1rem;
            border-bottom: 1px solid #eaeaea;
        }}
        
        .tab-button {{
            padding: 10px 20px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            font-size: 1rem;
            cursor: pointer;
            opacity: 0.7;
            transition: all 0.2s ease;
        }}
        
        .tab-button.active {{
            border-bottom-color: var(--apple-blue);
            opacity: 1;
            font-weight: 600;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Quantum vs Classical vs Hybrid Computing</h1>
        <p class="subtitle">Comprehensive Benchmark Results</p>
    </header>
    
    <div class="container">
        <div class="tab-container">
            <div class="tab-buttons">
                <button class="tab-button active" data-tab="dashboard">Dashboard</button>
                <button class="tab-button" data-tab="models">Model Comparison</button>
                <button class="tab-button" data-tab="paradigms">Computing Paradigms</button>
                <button class="tab-button" data-tab="visualizations">Visualizations</button>
            </div>
            
            <!-- Dashboard Tab -->
            <div class="tab-content active" id="dashboard">
                <h2>
                    <span class="icon">📊</span>
                    Performance Dashboard
                </h2>
                
                <p>This dashboard compares the performance of classical, quantum, and hybrid computing approaches across key metrics.</p>
                
                <div class="visualizations-grid">
                    <div class="visualization">
                        <div class="visualization-title">
                            <h3>Accuracy Comparison</h3>
                        </div>
                        <img src="accuracy_comparison.png" alt="Accuracy Comparison">
                    </div>
                    
                    <div class="visualization">
                        <div class="visualization-title">
                            <h3>Time Efficiency</h3>
                        </div>
                        <img src="time_comparison.png" alt="Time Comparison">
                    </div>
                    
                    <div class="visualization">
                        <div class="visualization-title">
                            <h3>Resource Usage</h3>
                        </div>
                        <img src="resource_usage.png" alt="Resource Usage">
                    </div>
                    
                    <div class="visualization">
                        <div class="visualization-title">
                            <h3>Multi-Dimensional View</h3>
                        </div>
                        <img src="radar_comparison.png" alt="Radar Comparison">
                    </div>
                </div>
                
                <h3>Top Performing Models</h3>
                
                <table class="models-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Training</th>
                            <th>Inference</th>
                            <th>Memory</th>
                            <th>Awards</th>
                        </tr>
                    </thead>
                    <tbody>
                        {model_rows}
                    </tbody>
                </table>
            </div>
            
            <!-- Models Tab -->
            <div class="tab-content" id="models">
                <h2>
                    <span class="icon">🧩</span>
                    Detailed Model Comparison
                </h2>
                
                <p>This section provides a detailed comparison of all models tested in the benchmark.</p>
                
                <img src="practicality_comparison.png" alt="Practicality Comparison" style="width: 100%; border-radius: 8px; margin: 1rem 0;">
                
                <table class="models-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Type</th>
                            <th>Accuracy</th>
                            <th>Training Time</th>
                            <th>Inference Time</th>
                            <th>Memory</th>
                            <th>Setup</th>
                            <th>Interpretability</th>
                        </tr>
                    </thead>
                    <tbody>
                        {model_rows.replace('<div class="model-type"', '<div class="model-type-hidden" style="display:none;"')}
                    </tbody>
                </table>
            </div>
            
            <!-- Paradigms Tab -->
            <div class="tab-content" id="paradigms">
                <h2>
                    <span class="icon">🔬</span>
                    Computing Paradigms
                </h2>
                
                <p>This section compares the three main computing paradigms: Classical, Quantum, and Hybrid approaches.</p>
                
                <div class="paradigms-comparison">
                    {type_comparison}
                </div>
                
                <h3 style="margin-top: 2rem;">Radar Comparison</h3>
                <p>This radar chart provides a multi-dimensional view of how each computing paradigm performs across key metrics.</p>
                
                <img src="radar_comparison.png" alt="Radar Comparison" style="width: 100%; max-width: 600px; margin: 0 auto; display: block; border-radius: 8px;">
            </div>
            
            <!-- Visualizations Tab -->
            <div class="tab-content" id="visualizations">
                <h2>
                    <span class="icon">📈</span>
                    All Visualizations
                </h2>
                
                <p>This section contains all the visualizations generated by the benchmark.</p>
                
                <div style="display: flex; flex-direction: column; gap: 2rem;">
                    <div>
                        <h3>Accuracy Comparison</h3>
                        <p>Higher percentages indicate better model performance.</p>
                        <img src="accuracy_comparison.png" alt="Accuracy Comparison" style="width: 100%; border-radius: 8px;">
                    </div>
                    
                    <div>
                        <h3>Training & Inference Time</h3>
                        <p>Lower times indicate faster model training and prediction.</p>
                        <img src="time_comparison.png" alt="Time Comparison" style="width: 100%; border-radius: 8px;">
                    </div>
                    
                    <div>
                        <h3>Resource Usage</h3>
                        <p>Lower resource usage indicates more efficient models.</p>
                        <img src="resource_usage.png" alt="Resource Usage" style="width: 100%; border-radius: 8px;">
                    </div>
                    
                    <div>
                        <h3>Practicality Metrics</h3>
                        <p>Higher scores indicate more practical, easier to implement models.</p>
                        <img src="practicality_comparison.png" alt="Practicality Comparison" style="width: 100%; border-radius: 8px;">
                    </div>
                    
                    <div>
                        <h3>Multi-Dimensional Comparison</h3>
                        <p>Radar chart showing performance across multiple dimensions simultaneously.</p>
                        <img src="radar_comparison.png" alt="Radar Comparison" style="width: 100%; border-radius: 8px;">
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>Quantum vs Classical vs Hybrid Computing Benchmark • Generated on {datetime.now().strftime("%Y-%m-%d")}</p>
    </footer>
    
    <script>
        // Tab functionality
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {{
            button.addEventListener('click', () => {{
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            }});
        }});
    </script>
</body>
</html>
    """
    
    # Save HTML dashboard
    with open(f"{OUTPUT_DIR}/benchmark_dashboard.html", 'w') as f:
        f.write(html_content)
    
    # Try to open the dashboard in browser
    try:
        dashboard_path = os.path.abspath(f"{OUTPUT_DIR}/benchmark_dashboard.html")
        print(f"\n🌟 Benchmark complete! Opening dashboard in your browser...")
        webbrowser.open(f"file://{dashboard_path}")
    except:
        print(f"\n🌟 Benchmark complete! Dashboard saved to {OUTPUT_DIR}/benchmark_dashboard.html")
    
    print(f"\n📂 All benchmark results saved to {OUTPUT_DIR}/")
    print(f"   • CSV data: all_results.csv, model_comparison.csv")
    print(f"   • Visualizations: accuracy, time, resource, practicality, radar charts")
    print(f"   • Interactive Dashboard: benchmark_dashboard.html")
    
    print("\n✨ To view the dashboard at any time, open this file in your browser:")
    print(f"   {os.path.abspath(f'{OUTPUT_DIR}/benchmark_dashboard.html')}")

if __name__ == "__main__":
    run_ultimate_benchmark() 