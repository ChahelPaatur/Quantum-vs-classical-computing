"""
Visualization utilities for displaying performance metrics and model comparisons
with Apple-inspired design aesthetics.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


# Define Apple-inspired color palette
APPLE_COLORS = {
    'blue': '#007AFF',  # iOS blue
    'green': '#34C759',  # iOS green
    'indigo': '#5856D6',  # iOS indigo
    'orange': '#FF9500',  # iOS orange
    'pink': '#FF2D55',  # iOS pink
    'purple': '#AF52DE',  # iOS purple
    'red': '#FF3B30',  # iOS red
    'teal': '#5AC8FA',  # iOS teal
    'yellow': '#FFCC00',  # iOS yellow
    'gold': '#FFD700',   # Gold color for Hybrid models
    'amber': '#FFBF00',  # Amber alternative
    'gray': '#8E8E93',  # iOS gray
    'light_gray': '#E5E5EA', # iOS light gray
    'white': '#FFFFFF',  # White
    'black': '#000000',  # Black
    'background': '#F2F2F7'  # iOS light background
}

# Create a custom colormap based on Apple's gradient style
APPLE_CMAP = LinearSegmentedColormap.from_list(
    'apple_gradient', 
    [APPLE_COLORS['blue'], APPLE_COLORS['purple'], APPLE_COLORS['pink']]
)


class AppleStyleVisualizer:
    """Visualizer class with Apple-inspired design aesthetics."""
    
    def __init__(self):
        """Initialize the visualizer with Apple-inspired styling."""
        self.setup_style()
        
    def setup_style(self):
        """Set up matplotlib styling to match Apple aesthetics."""
        # Set the font to a sans-serif font similar to San Francisco
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SF Pro Display', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']
        
        # Set background colors and grid styles
        plt.rcParams['figure.facecolor'] = APPLE_COLORS['background']
        plt.rcParams['axes.facecolor'] = APPLE_COLORS['white']
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = APPLE_COLORS['light_gray']
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 0.5
        
        # Set text colors
        plt.rcParams['text.color'] = APPLE_COLORS['black']
        plt.rcParams['axes.labelcolor'] = APPLE_COLORS['black']
        plt.rcParams['xtick.color'] = APPLE_COLORS['black']
        plt.rcParams['ytick.color'] = APPLE_COLORS['black']
        
        # Set line styles
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
            APPLE_COLORS['blue'], 
            APPLE_COLORS['green'], 
            APPLE_COLORS['indigo'],
            APPLE_COLORS['orange'],
            APPLE_COLORS['pink'],
            APPLE_COLORS['purple'],
            APPLE_COLORS['red'],
            APPLE_COLORS['teal'],
            APPLE_COLORS['yellow']
        ])
        
        # Set figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Set thicker lines
        plt.rcParams['lines.linewidth'] = 2.5
        
        # Set font sizes
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 14
        
        # Configure seaborn
        sns.set_style("whitegrid")
        sns.set_palette([
            APPLE_COLORS['blue'], 
            APPLE_COLORS['green'], 
            APPLE_COLORS['indigo']
        ])
    
    def plot_accuracy_comparison(self, results_dict, metric='accuracy', title='Model Accuracy Comparison', 
                                 figsize=(12, 8), save_path=None):
        """
        Create a bar chart comparing model accuracy.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics dictionaries as values
            metric (str): Metric to plot (default: 'accuracy')
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        plt.figure(figsize=figsize)
        
        # Extract model names and values
        models = []
        values = []
        
        for model_name, metrics in results_dict.items():
            if metric in metrics:
                models.append(model_name)
                values.append(metrics[metric])
        
        # Create bar colors
        colors = [APPLE_COLORS['blue'], APPLE_COLORS['purple'], APPLE_COLORS['pink']]
        
        # Create plot
        bars = plt.bar(models, values, color=colors, width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.4f}',
                ha='center', 
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        plt.title(title, fontsize=20, fontweight='bold', pad=20)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add subtle rounded background
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(APPLE_COLORS['light_gray'])
        ax.spines['bottom'].set_color(APPLE_COLORS['light_gray'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_time_comparison(self, results_dict, figsize=(12, 8), save_path=None):
        """
        Create a bar chart comparing model execution times.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics dictionaries as values
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        plt.figure(figsize=figsize)
        
        # Extract model names and execution times
        models = []
        times = []
        
        for model_name, metrics in results_dict.items():
            if 'execution_time' in metrics:
                models.append(model_name)
                times.append(metrics['execution_time'])
        
        # Create bar colors
        colors = [APPLE_COLORS['green'], APPLE_COLORS['orange'], APPLE_COLORS['red']]
        
        # Create plot
        bars = plt.bar(models, times, color=colors, width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.2f}s',
                ha='center', 
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        plt.title('Model Execution Time Comparison', fontsize=20, fontweight='bold', pad=20)
        plt.ylabel('Execution Time (seconds)', fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add subtle rounded background
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(APPLE_COLORS['light_gray'])
        ax.spines['bottom'].set_color(APPLE_COLORS['light_gray'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_resource_usage(self, results_dict, figsize=(12, 9), save_path=None):
        """
        Create a plot comparing CPU and memory usage.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics dictionaries as values
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Determine which metrics are available
        sample_model = next(iter(results_dict.values()))
        cpu_field = None
        memory_field = None
        
        # Check for CPU metrics in order of preference
        for field in ['avg_cpu', 'cpu_usage', 'max_cpu']:
            if field in sample_model:
                cpu_field = field
                break
                
        # Check for memory metrics in order of preference
        for field in ['avg_memory', 'memory_usage', 'max_memory']:
            if field in sample_model:
                memory_field = field
                break
        
        if not cpu_field or not memory_field:
            print(f"Warning: Missing required metrics for resource usage plot. Available metrics: {list(sample_model.keys())}")
            return fig
        
        # Extract model names, CPU and memory usage
        models = []
        model_types = []
        cpu_usage_values = []
        memory_usage_values = []
        
        for model_name, metrics in results_dict.items():
            models.append(model_name)
            model_types.append(metrics['model_type'])
            cpu_usage_values.append(metrics[cpu_field])
            memory_usage_values.append(metrics[memory_field])
        
        # Create bar colors based on model type
        colors = []
        for model_type in model_types:
            if model_type == 'Classical':
                colors.append(APPLE_COLORS['blue'])
            elif model_type == 'Quantum':
                colors.append(APPLE_COLORS['green'])
            else:  # Hybrid
                colors.append(APPLE_COLORS['amber'])  # Using amber instead of gold
        
        # Plot CPU usage
        bars1 = ax1.bar(models, cpu_usage_values, color=colors, width=0.6)
        ax1.set_title(f'CPU Usage Comparison ({cpu_field.replace("_", " ").title()})', 
                     fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel('CPU Usage (%)', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.1f}%',
                ha='center', 
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Plot Memory usage
        bars2 = ax2.bar(models, memory_usage_values, color=colors, width=0.6)
        ax2.set_title(f'Memory Usage Comparison ({memory_field.replace("_", " ").title()})', 
                     fontsize=16, fontweight='bold', pad=15)
        ax2.set_ylabel('Memory Used (MB)', fontsize=12)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.1f} MB',
                ha='center', 
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Set y-axes to start at 0
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        # Add subtle rounded background
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(APPLE_COLORS['light_gray'])
            ax.spines['bottom'].set_color(APPLE_COLORS['light_gray'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_radar_comparison(self, results_dict, metrics, figsize=(10, 10), save_path=None):
        """
        Create a radar chart comparing multiple metrics across models.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics dictionaries as values
            metrics (list): List of metrics to include
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Number of metrics
        N = len(metrics)
        
        # Set up the angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Make the plot circular
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Define which metrics are better when higher (True) or lower (False)
        better_when_higher = {
            'accuracy': True,
            'precision': True,
            'recall': True,
            'f1_score': True,
            'roc_auc': True,
            'r2': True,
            'practicality_score': True,
            'setup_difficulty': True,
            'interpretability': True,
            'deployment_complexity': True,
            'hardware_requirements': True,
            'scalability': True,
            'training_time': False,
            'inference_time': False,
            'memory_usage': False,
            'cpu_usage': False,
            'max_memory': False,
            'avg_memory': False,
            'max_cpu': False,
            'avg_cpu': False,
        }
        
        # Extract normalized values for each model
        normalized_values = {}
        metrics_extended = metrics + [metrics[0]]
        
        # Group models by type
        models_by_type = {}
        for model, metrics_dict in results_dict.items():
            model_type = metrics_dict['model_type']
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append(model)
        
        # Calculate average values by model type
        type_values = {}
        for model_type, model_list in models_by_type.items():
            type_values[model_type] = []
            
            for metric in metrics:
                # Get values for this metric from all models of this type
                values = [results_dict[model][metric] for model in model_list]
                # Calculate the average
                avg_value = np.mean(values)
                type_values[model_type].append(avg_value)
        
        # Normalize the values
        for metric_idx, metric in enumerate(metrics):
            max_value = max(tv[metric_idx] for tv in type_values.values())
            min_value = min(tv[metric_idx] for tv in type_values.values())
            
            for model_type in type_values:
                value = type_values[model_type][metric_idx]
                
                # Check if higher is better for this metric
                higher_is_better = better_when_higher.get(metric, True)
                
                if max_value == min_value:
                    normalized = 0.5  # Default if all values are the same
                else:
                    # Normalize to 0-1 range
                    normalized = (value - min_value) / (max_value - min_value)
                    
                    # Invert if lower is better
                    if not higher_is_better:
                        normalized = 1 - normalized
                
                type_values[model_type][metric_idx] = normalized
        
        # Complete the loop for each model type
        for model_type in type_values:
            type_values[model_type].append(type_values[model_type][0])
        
        # Plot each model type
        color_map = {
            'Classical': APPLE_COLORS['blue'],
            'Quantum': APPLE_COLORS['green'],
            'Hybrid': APPLE_COLORS['gold']
        }
        
        for model_type, values in type_values.items():
            color = color_map.get(model_type, APPLE_COLORS['purple'])
            ax.plot(angles, values, linewidth=3, linestyle='solid', color=color, label=model_type)
            ax.fill(angles, values, color=color, alpha=0.2)
        
        # Set the labels
        metric_labels = [metric.replace('_', ' ').title() for metric in metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
        
        # Remove radial labels and add light grid
        ax.set_yticklabels([])
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.2, 0.1), frameon=True, fontsize=12, 
                  edgecolor=APPLE_COLORS['light_gray'])
        
        plt.title('Multi-Dimensional Performance Comparison', fontsize=20, fontweight='bold', y=1.08)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_practicality_comparison(self, results_dict, figsize=(14, 7), save_path=None):
        """
        Create a grouped bar chart comparing practicality metrics.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics dictionaries as values
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        plt.figure(figsize=figsize)
        
        # Extract model names and practicality metrics
        models = []
        metrics = ['setup_difficulty', 'interpretability', 'deployment_complexity', 
                   'hardware_requirements', 'scalability']
        metric_labels = ['Setup Ease', 'Interpretability', 'Deployment Ease', 
                         'Hardware Accessibility', 'Scalability']
        data = {metric: [] for metric in metrics}
        
        for model_name, model_metrics in results_dict.items():
            # Support both nested and direct structure for practicality metrics
            if 'practicality' in model_metrics:
                # Nested structure (practicality is a dictionary containing metrics)
                models.append(model_name)
                for metric in metrics:
                    data[metric].append(model_metrics['practicality'][metric])
            elif all(metric in model_metrics for metric in metrics):
                # Direct structure (metrics are direct keys in the model_metrics dict)
                models.append(model_name)
                for metric in metrics:
                    data[metric].append(model_metrics[metric])
        
        # Only proceed if we have data to plot
        if not models:
            print("Warning: No practicality metrics found in the data. Skipping practicality comparison plot.")
            empty_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No practicality metrics data available", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return empty_fig
        
        # Set width of bars
        barWidth = 0.15
        
        # Set position of bars on X axis
        r = np.arange(len(models))
        positions = [r]
        for i in range(1, len(metrics)):
            positions.append([x + barWidth for x in positions[-1]])
        
        # Create colors
        colors = [APPLE_COLORS['blue'], APPLE_COLORS['green'], APPLE_COLORS['indigo'],
                 APPLE_COLORS['orange'], APPLE_COLORS['purple']]
        
        # Make the plot
        for i, metric in enumerate(metrics):
            plt.bar(positions[i], data[metric], width=barWidth, color=colors[i % len(colors)], 
                   edgecolor='white', label=metric_labels[i])
        
        # Add xticks on the middle of the group bars
        plt.xlabel('Models', fontsize=14, fontweight='bold')
        plt.ylabel('Score (Higher is Better)', fontsize=14, fontweight='bold')
        plt.xticks([r + barWidth * 2 for r in range(len(models))], models, rotation=45, ha='right')
        
        # Create legend & title
        plt.title('Practicality Metrics Comparison', fontsize=20, fontweight='bold', pad=20)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        
        # Set y-axis limits
        plt.ylim(0, 10.5)
        plt.yticks(range(0, 11, 1))
        
        # Add subtle rounded background
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(APPLE_COLORS['light_gray'])
        ax.spines['bottom'].set_color(APPLE_COLORS['light_gray'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_comprehensive_dashboard(self, results_dict, task_type='classification', 
                                    figsize=(16, 12), save_path=None):
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics dictionaries as values
            task_type (str): Type of task ('classification' or 'regression')
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3)
        
        # Determine which metrics to use based on task_type
        if task_type == 'classification':
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
            main_metric = 'accuracy'
            main_title = 'Accuracy Comparison'
        else:  # regression
            performance_metrics = ['mse', 'rmse', 'mae', 'r2']
            main_metric = 'r2'
            main_title = 'R² Score Comparison'
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, :])  # Main metric bar chart
        ax2 = fig.add_subplot(gs[1, 0])  # Time comparison
        ax3 = fig.add_subplot(gs[1, 1])  # CPU usage
        ax4 = fig.add_subplot(gs[1, 2])  # Memory usage
        ax5 = fig.add_subplot(gs[2, :2], polar=True)  # Radar chart
        ax6 = fig.add_subplot(gs[2, 2])  # Practicality summary
        
        # 1. Main metric bar chart
        models = []
        values = []
        for model_name, metrics in results_dict.items():
            if main_metric in metrics:
                models.append(model_name)
                values.append(metrics[main_metric])
        
        colors = [APPLE_COLORS['blue'], APPLE_COLORS['purple'], APPLE_COLORS['pink']]
        bars = ax1.bar(models, values, color=colors, width=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.4f}',
                ha='center', 
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax1.set_title(main_title, fontsize=16, fontweight='bold')
        ax1.set_ylim(bottom=0)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. Time comparison
        times = []
        for model_name in models:
            if 'execution_time' in results_dict[model_name]:
                times.append(results_dict[model_name]['execution_time'])
            else:
                times.append(0)
        
        bars = ax2.bar(models, times, color=colors, width=0.6)
        ax2.set_title('Execution Time (s)', fontsize=14, fontweight='bold')
        ax2.set_ylim(bottom=0)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. CPU usage
        cpu_values = []
        for model_name in models:
            if 'avg_cpu' in results_dict[model_name]:
                cpu_values.append(results_dict[model_name]['avg_cpu'])
            else:
                cpu_values.append(0)
        
        bars = ax3.bar(models, cpu_values, color=colors, width=0.6)
        ax3.set_title('CPU Usage (%)', fontsize=14, fontweight='bold')
        ax3.set_ylim(bottom=0)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Memory usage
        memory_values = []
        for model_name in models:
            if 'memory_used' in results_dict[model_name]:
                memory_values.append(results_dict[model_name]['memory_used'])
            else:
                memory_values.append(0)
        
        bars = ax4.bar(models, memory_values, color=colors, width=0.6)
        ax4.set_title('Memory Usage (MB)', fontsize=14, fontweight='bold')
        ax4.set_ylim(bottom=0)
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 5. Radar chart for combined metrics
        metrics_to_plot = performance_metrics + ['execution_time']
        
        # Number of metrics
        N = len(metrics_to_plot)
        
        # Set up the angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Make the plot circular
        angles += angles[:1]
        
        # Extract normalized values for each model
        normalized_values = {}
        
        for metric in metrics_to_plot:
            max_values = [results_dict[model].get(metric, 0) for model in models]
            if not any(max_values):
                continue
                
            max_value = max(results_dict[model].get(metric, 0) for model in models)
            min_value = min(results_dict[model].get(metric, 0) for model in models)
            
            for model in models:
                if model not in normalized_values:
                    normalized_values[model] = []
                
                if metric in results_dict[model]:
                    # Check if higher is better for this metric
                    higher_is_better = metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2']
                    
                    if max_value == min_value:
                        normalized = 0.5  # Default if all values are the same
                    else:
                        value = results_dict[model].get(metric, 0)
                        normalized = (value - min_value) / (max_value - min_value)
                        
                        if not higher_is_better:
                            normalized = 1 - normalized
                    
                    normalized_values[model].append(normalized)
                else:
                    normalized_values[model].append(0)
        
        # Complete the loop for each model
        for model in normalized_values:
            normalized_values[model] += normalized_values[model][:1]
            
        # Plot each model
        for i, (model, values) in enumerate(normalized_values.items()):
            if len(values) < 2:  # Skip if not enough data
                continue
            ax5.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i % len(colors)], label=model)
            ax5.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
        
        # Set the labels
        metric_labels = [metric.replace('_', ' ').title() for metric in metrics_to_plot]
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metric_labels, fontsize=10)
        
        # Remove radial labels
        ax5.set_yticklabels([])
        ax5.set_title('Combined Performance', fontsize=14, fontweight='bold')
        
        # 6. Practicality summary
        practicality_data = {}
        for model in models:
            if 'practicality' in results_dict[model]:
                practicality_data[model] = results_dict[model]['practicality']['overall_practicality']
            else:
                practicality_data[model] = 0
        
        bars = ax6.bar(practicality_data.keys(), practicality_data.values(), color=colors, width=0.6)
        ax6.set_title('Overall Practicality', fontsize=14, fontweight='bold')
        ax6.set_ylim(0, 10)
        ax6.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.1f}',
                ha='center', 
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Add a title to the figure
        fig.suptitle('Comprehensive Model Comparison Dashboard', fontsize=24, fontweight='bold', y=0.98)
        
        # Add subtle styling
        for ax in [ax1, ax2, ax3, ax4, ax6]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(APPLE_COLORS['light_gray'])
            ax.spines['bottom'].set_color(APPLE_COLORS['light_gray'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig 