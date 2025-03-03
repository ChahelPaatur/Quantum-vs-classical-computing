"""
Utilities for measuring and comparing performance metrics across different computing models.
"""
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


class ModelEvaluator:
    """Class for evaluating and comparing different computing models."""
    
    def __init__(self):
        self.results = {}
        
    def measure_time(self, func, *args, **kwargs):
        """
        Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            tuple: (result of function, execution time in seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def measure_power_usage(self, func, *args, **kwargs):
        """
        Estimate power usage by measuring CPU utilization.
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            tuple: (result of function, avg CPU usage %, max memory usage MB)
        """
        # Get baseline
        process = psutil.Process()
        baseline_cpu = process.cpu_percent(interval=0.1)
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Initialize tracking variables
        cpu_measurements = []
        memory_measurements = []
        
        # Define monitoring function
        def monitor_resources():
            while not monitoring_complete[0]:
                cpu_measurements.append(process.cpu_percent(interval=0.1))
                memory_measurements.append(process.memory_info().rss / (1024 * 1024))
                time.sleep(0.1)
        
        # Set up monitoring in a separate thread
        import threading
        monitoring_complete = [False]
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Stop monitoring
        monitoring_complete[0] = True
        monitor_thread.join(timeout=1.0)
        
        # Calculate metrics
        avg_cpu = np.mean(cpu_measurements) if cpu_measurements else 0
        max_memory = np.max(memory_measurements) if memory_measurements else 0
        memory_used = max_memory - baseline_memory
        
        return result, avg_cpu, memory_used
    
    def evaluate_classification_model(self, model_name, y_true, y_pred, y_proba=None, execution_time=None, power_metrics=None):
        """
        Evaluate a classification model and store results.
        
        Args:
            model_name (str): Name of the model
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_proba (array, optional): Predicted probabilities for ROC AUC
            execution_time (float, optional): Model execution time
            power_metrics (tuple, optional): (avg_cpu, memory_used)
        """
        metrics = {}
        
        # Accuracy metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC if probabilities are provided
        if y_proba is not None:
            try:
                if y_proba.shape[1] > 2:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                else:  # Binary
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['roc_auc'] = None
        
        # Time and power metrics
        if execution_time is not None:
            metrics['execution_time'] = execution_time
        
        if power_metrics is not None:
            metrics['avg_cpu'] = power_metrics[0]
            metrics['memory_used'] = power_metrics[1]
        
        # Store detailed classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # Store confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_regression_model(self, model_name, y_true, y_pred, execution_time=None, power_metrics=None):
        """
        Evaluate a regression model and store results.
        
        Args:
            model_name (str): Name of the model
            y_true (array): True values
            y_pred (array): Predicted values
            execution_time (float, optional): Model execution time
            power_metrics (tuple, optional): (avg_cpu, memory_used)
        """
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Time and power metrics
        if execution_time is not None:
            metrics['execution_time'] = execution_time
        
        if power_metrics is not None:
            metrics['avg_cpu'] = power_metrics[0]
            metrics['memory_used'] = power_metrics[1]
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_forecasting(self, model_name, y_true, y_pred, horizon=1, execution_time=None, power_metrics=None):
        """
        Evaluate a forecasting model and store results.
        
        Args:
            model_name (str): Name of the model
            y_true (array): True values
            y_pred (array): Predicted values
            horizon (int): Forecasting horizon
            execution_time (float, optional): Model execution time
            power_metrics (tuple, optional): (avg_cpu, memory_used)
        """
        metrics = {}
        
        # Forecasting metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0  # Avoid division by zero
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = None
            
        # Time and power metrics
        if execution_time is not None:
            metrics['execution_time'] = execution_time
        
        if power_metrics is not None:
            metrics['avg_cpu'] = power_metrics[0]
            metrics['memory_used'] = power_metrics[1]
        
        # Store horizon information
        metrics['horizon'] = horizon
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_practicality(self, model_name, setup_difficulty=1, interpretability=1, 
                              deployment_complexity=1, hardware_requirements=1, scalability=1):
        """
        Evaluate the practicality of a model based on qualitative factors.
        Each factor is rated on a scale of 1-10, with 10 being best.
        
        Args:
            model_name (str): Name of the model
            setup_difficulty (int): Ease of setup (1-10)
            interpretability (int): Model interpretability (1-10)
            deployment_complexity (int): Ease of deployment (1-10)
            hardware_requirements (int): Hardware accessibility (1-10)
            scalability (int): Ability to scale to larger problems (1-10)
        """
        if model_name not in self.results:
            self.results[model_name] = {}
            
        practicality_metrics = {
            'setup_difficulty': setup_difficulty,
            'interpretability': interpretability,
            'deployment_complexity': deployment_complexity,
            'hardware_requirements': hardware_requirements,
            'scalability': scalability,
            'overall_practicality': np.mean([
                setup_difficulty, 
                interpretability, 
                deployment_complexity,
                hardware_requirements,
                scalability
            ])
        }
        
        self.results[model_name]['practicality'] = practicality_metrics
        return practicality_metrics
    
    def get_comparative_metrics(self):
        """
        Generate a comparative summary of all evaluated models.
        
        Returns:
            dict: Comparative metrics
        """
        if not self.results:
            return {}
        
        comparative = {
            'model_names': list(self.results.keys()),
            'metrics': {}
        }
        
        # Check if we have classification or regression models
        first_model = next(iter(self.results.values()))
        is_classification = 'accuracy' in first_model
        
        # Common metrics
        common_metrics = ['execution_time', 'avg_cpu', 'memory_used']
        
        # Add model-specific metrics
        if is_classification:
            task_metrics = ['accuracy', 'precision', 'recall', 'f1']
        else:
            task_metrics = ['mse', 'rmse', 'mae', 'r2']
        
        all_metrics = task_metrics + common_metrics
        
        # Collect metrics
        for metric in all_metrics:
            if all(metric in model for model in self.results.values()):
                comparative['metrics'][metric] = {
                    name: results[metric] 
                    for name, results in self.results.items()
                }
        
        # Add practicality if available
        if all('practicality' in model for model in self.results.values()):
            comparative['metrics']['overall_practicality'] = {
                name: results['practicality']['overall_practicality'] 
                for name, results in self.results.items()
            }
        
        return comparative 