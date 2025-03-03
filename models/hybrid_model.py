"""
Hybrid machine learning model implementations that combine classical and quantum computing.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM

# Pennylane imports
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

# Import from our other models
from models.classical_model import ClassicalModel
from models.quantum_model import QuantumModel


class HybridModel:
    """Base class for hybrid machine learning models that combine classical and quantum approaches."""
    
    def __init__(self, hybrid_type='feature_hybrid', classical_model_type='random_forest', 
                 quantum_framework='qiskit', quantum_model_type='variational',
                 task_type='classification', n_qubits=None, quantum_percentage=0.3):
        """
        Initialize a hybrid model.
        
        Args:
            hybrid_type (str): Type of hybrid approach ('feature_hybrid', 'ensemble', 'quantum_enhanced')
            classical_model_type (str): Type of classical model
            quantum_framework (str): Quantum framework to use ('qiskit' or 'pennylane')
            quantum_model_type (str): Type of quantum model
            task_type (str): Type of task ('classification' or 'regression')
            n_qubits (int): Number of qubits to use
            quantum_percentage (float): Percentage of features/models to be processed by quantum model (0-1)
        """
        self.hybrid_type = hybrid_type
        self.classical_model_type = classical_model_type
        self.quantum_framework = quantum_framework
        self.quantum_model_type = quantum_model_type
        self.task_type = task_type
        self.n_qubits = n_qubits
        self.quantum_percentage = quantum_percentage
        
        # Initialize models
        self.classical_model = None
        self.quantum_model = None
        self.feature_selector = None
        self.pca = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        # For ensemble method
        self.models = []
        self.weights = []
        
    def _create_feature_hybrid(self):
        """
        Create a feature-based hybrid model where some features are processed 
        classically and others quantum mechanically.
        """
        # Create classical model
        self.classical_model = ClassicalModel(
            model_type=self.classical_model_type, 
            task_type=self.task_type
        )
        
        # Create quantum model
        self.quantum_model = QuantumModel(
            framework=self.quantum_framework,
            model_type=self.quantum_model_type,
            task_type=self.task_type,
            n_qubits=self.n_qubits
        )
    
    def _create_ensemble(self):
        """
        Create an ensemble hybrid model that combines predictions from 
        classical and quantum models.
        """
        # Create multiple models with different configurations
        # 1. Classical models
        models = [
            ('classical_rf', ClassicalModel('random_forest', self.task_type)),
            ('classical_gb', ClassicalModel('gradient_boosting', self.task_type)),
            ('classical_nn', ClassicalModel('neural_network', self.task_type))
        ]
        
        # 2. Quantum models
        if self.n_qubits is not None and self.n_qubits <= 4:  # Only add if qubit count is reasonable
            models.extend([
                ('quantum_qiskit', QuantumModel('qiskit', 'variational', self.task_type, self.n_qubits)),
                ('quantum_pennylane', QuantumModel('pennylane', 'variational', self.task_type, self.n_qubits))
            ])
        
        self.models = models
        # Start with equal weights, will be adjusted during training
        self.weights = np.ones(len(models)) / len(models)
    
    def _create_quantum_enhanced(self):
        """
        Create a quantum-enhanced classical model where quantum computing
        is used to enhance feature extraction or model selection.
        """
        # Create a primary classical model
        self.classical_model = ClassicalModel(
            model_type=self.classical_model_type, 
            task_type=self.task_type
        )
        
        # Create quantum feature circuit for feature extraction
        self.quantum_model = QuantumModel(
            framework=self.quantum_framework,
            model_type=self.quantum_model_type,
            task_type='classification',  # Always classification for feature importance
            n_qubits=min(8, self.n_qubits) if self.n_qubits else 4
        )
        
        # Create PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
    
    def _select_features(self, X, n_quantum_features=None):
        """
        Select which features to process quantum mechanically.
        
        Args:
            X (array-like): Input features
            n_quantum_features (int, optional): Number of features to select
            
        Returns:
            tuple: (quantum_features, classical_features, quantum_indices, classical_indices)
        """
        if n_quantum_features is None:
            n_quantum_features = max(1, int(X.shape[1] * self.quantum_percentage))
        
        n_quantum_features = min(n_quantum_features, X.shape[1])
        n_classical_features = X.shape[1] - n_quantum_features
        
        # Use PCA to identify the most important features
        if self.feature_selector is None:
            # Use PCA to rank features by importance
            self.feature_selector = PCA(n_components=X.shape[1])
            X_scaled = self.feature_scaler.fit_transform(X)
            self.feature_selector.fit(X_scaled)
            
            # Get feature importance based on PCA components
            components = np.abs(self.feature_selector.components_)
            feature_importance = np.sum(components, axis=0)
            self.feature_ranking = np.argsort(feature_importance)[::-1]  # Descending order
        
        # Select quantum and classical features
        quantum_indices = self.feature_ranking[:n_quantum_features]
        classical_indices = self.feature_ranking[n_quantum_features:]
        
        # Extract features
        quantum_features = X[:, quantum_indices]
        classical_features = X[:, classical_indices] if n_classical_features > 0 else None
        
        return quantum_features, classical_features, quantum_indices, classical_indices
    
    def _quantum_feature_extraction(self, X):
        """
        Use quantum circuits to extract new features from input data.
        
        Args:
            X (array-like): Input features
            
        Returns:
            array: Quantum-enhanced features
        """
        # Scale the input data
        X_scaled = self.feature_scaler.transform(X)
        
        # Use PCA to reduce dimensions if needed
        if self.pca is not None:
            X_reduced = self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Use the quantum model to extract features
        # This depends on the framework
        if self.quantum_model.framework == 'qiskit':
            # If the model is already fitted, use it
            if hasattr(self.quantum_model, 'is_fitted') and self.quantum_model.is_fitted:
                # For Qiskit, we'll use the internal feature map to extract features
                if hasattr(self.quantum_model.model, 'feature_map'):
                    # Get the feature map
                    feature_map = self.quantum_model.model.feature_map
                    
                    # Create a quantum circuit for each sample and extract features
                    backend = Aer.get_backend('statevector_simulator')
                    quantum_features = []
                    
                    for x in X_reduced:
                        # Prepare a circuit with the feature map
                        qc = QuantumCircuit(feature_map.num_qubits)
                        feature_map_params = {}
                        qc = feature_map.bind_parameters(feature_map_params)
                        
                        # Execute the circuit and get the statevector
                        job = execute(qc, backend)
                        statevector = job.result().get_statevector()
                        
                        # Use the real part of the statevector as features
                        features = np.real(statevector)
                        quantum_features.append(features)
                    
                    return np.array(quantum_features)
                else:
                    # If no feature map, just return the original data
                    return X_reduced
            else:
                # If not fitted, just return the original data
                return X_reduced
        
        elif self.quantum_model.framework == 'pennylane':
            # If the model is already fitted, use it
            if hasattr(self.quantum_model, 'is_fitted') and self.quantum_model.is_fitted:
                # For PennyLane, we'll create a new circuit to extract features
                if hasattr(self.quantum_model, 'device') and self.quantum_model.device is not None:
                    # Create a new qnode for feature extraction
                    dev = self.quantum_model.device
                    
                    @qml.qnode(dev)
                    def feature_circuit(x):
                        # Embed the input
                        AngleEmbedding(x, wires=range(dev.num_wires))
                        
                        # Return the expectation values of all qubits
                        return [qml.expval(qml.PauliZ(i)) for i in range(dev.num_wires)]
                    
                    # Extract features for each sample
                    quantum_features = []
                    for x in X_reduced:
                        features = feature_circuit(x[:dev.num_wires])
                        quantum_features.append(features)
                    
                    return np.array(quantum_features)
                else:
                    # If no device, just return the original data
                    return X_reduced
            else:
                # If not fitted, just return the original data
                return X_reduced
        
        else:
            # For unsupported frameworks, just return the original data
            return X_reduced
    
    def fit(self, X, y):
        """
        Train the hybrid model.
        
        Args:
            X (array-like): Training features
            y (array-like): Target values
            
        Returns:
            HybridModel: Trained model instance
        """
        # Create models if not already created
        if self.hybrid_type == 'feature_hybrid':
            if self.classical_model is None or self.quantum_model is None:
                self._create_feature_hybrid()
                
            # Select features for classical and quantum models
            quantum_features, classical_features, _, _ = self._select_features(X)
            
            # Train the quantum model with selected features
            if quantum_features is not None:
                self.quantum_model.fit(quantum_features, y)
            
            # Train the classical model with selected features
            if classical_features is not None:
                self.classical_model.fit(classical_features, y)
            
        elif self.hybrid_type == 'ensemble':
            if not self.models:
                self._create_ensemble()
            
            # Train each model in the ensemble
            trained_models = []
            accuracies = []
            
            for name, model in self.models:
                print(f"Training ensemble model: {name}")
                model.fit(X, y)
                trained_models.append((name, model))
                
                # Estimate model performance with cross-validation
                if self.task_type == 'classification':
                    from sklearn.metrics import accuracy_score
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    fold_accuracies = []
                    
                    for train_idx, val_idx in kf.split(X):
                        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                        
                        model_fold = type(model)(
                            model_type=model.model_type if hasattr(model, 'model_type') else None,
                            task_type=self.task_type
                        )
                        model_fold.fit(X_fold_train, y_fold_train)
                        y_fold_pred = model_fold.predict(X_fold_val)
                        
                        fold_accuracies.append(accuracy_score(y_fold_val, y_fold_pred))
                    
                    model_accuracy = np.mean(fold_accuracies)
                    accuracies.append(model_accuracy)
                    print(f"  Cross-validation accuracy: {model_accuracy:.4f}")
                else:
                    from sklearn.metrics import r2_score
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    fold_r2s = []
                    
                    for train_idx, val_idx in kf.split(X):
                        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                        
                        model_fold = type(model)(
                            model_type=model.model_type if hasattr(model, 'model_type') else None,
                            task_type=self.task_type
                        )
                        model_fold.fit(X_fold_train, y_fold_train)
                        y_fold_pred = model_fold.predict(X_fold_val)
                        
                        fold_r2s.append(r2_score(y_fold_val, y_fold_pred))
                    
                    model_r2 = np.mean(fold_r2s)
                    accuracies.append(model_r2)
                    print(f"  Cross-validation R²: {model_r2:.4f}")
            
            # Update model weights based on performance
            if accuracies:
                accuracies = np.array(accuracies)
                # Apply softmax to get model weights
                self.weights = np.exp(accuracies) / np.sum(np.exp(accuracies))
                
            # Save the trained models
            self.models = trained_models
            
        elif self.hybrid_type == 'quantum_enhanced':
            if self.classical_model is None or self.quantum_model is None:
                self._create_quantum_enhanced()
            
            # First, extract quantum-enhanced features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Fit the PCA if needed
            if self.pca is not None:
                X_reduced = self.pca.fit_transform(X_scaled)
            else:
                X_reduced = X_scaled
            
            # Fit the quantum model for feature extraction
            try:
                self.quantum_model.fit(X_reduced[:100], y[:100])  # Using a subset for efficiency
            except Exception as e:
                print(f"Warning: Quantum model fitting failed: {e}")
                
            # Extract quantum features
            quantum_features = self._quantum_feature_extraction(X)
            
            # Combine original features with quantum features
            X_enhanced = np.hstack([X, quantum_features]) if quantum_features is not None else X
            
            # Train the classical model with enhanced features
            self.classical_model.fit(X_enhanced, y)
        
        else:
            raise ValueError(f"Unsupported hybrid type: {self.hybrid_type}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained hybrid model.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        if self.hybrid_type == 'feature_hybrid':
            # Select features for classical and quantum models
            quantum_features, classical_features, _, _ = self._select_features(X)
            
            # Get predictions from quantum model
            quantum_pred = self.quantum_model.predict(quantum_features) if quantum_features is not None else None
            
            # Get predictions from classical model
            classical_pred = self.classical_model.predict(classical_features) if classical_features is not None else None
            
            # Combine predictions
            if quantum_pred is not None and classical_pred is not None:
                if self.task_type == 'classification':
                    # For classification, use voting
                    # Convert both predictions to one-hot encoding
                    from sklearn.preprocessing import OneHotEncoder
                    
                    # Handle different shapes of predictions
                    quantum_pred = quantum_pred.reshape(-1, 1) if quantum_pred.ndim == 1 else quantum_pred
                    classical_pred = classical_pred.reshape(-1, 1) if classical_pred.ndim == 1 else classical_pred
                    
                    ohe = OneHotEncoder(sparse=False)
                    ohe.fit(np.vstack([quantum_pred, classical_pred]))
                    
                    quantum_one_hot = ohe.transform(quantum_pred)
                    classical_one_hot = ohe.transform(classical_pred)
                    
                    # Weight the predictions by the quantum percentage
                    combined_proba = self.quantum_percentage * quantum_one_hot + \
                                    (1 - self.quantum_percentage) * classical_one_hot
                    
                    # Get the class with highest probability
                    return np.argmax(combined_proba, axis=1)
                else:
                    # For regression, use weighted average
                    return self.quantum_percentage * quantum_pred + \
                           (1 - self.quantum_percentage) * classical_pred
            elif quantum_pred is not None:
                return quantum_pred
            elif classical_pred is not None:
                return classical_pred
            else:
                raise RuntimeError("Both quantum and classical models failed to predict")
                
        elif self.hybrid_type == 'ensemble':
            # Get predictions from all models
            all_predictions = []
            for _, model in self.models:
                try:
                    preds = model.predict(X)
                    all_predictions.append(preds)
                except Exception as e:
                    print(f"Warning: Model prediction failed: {e}")
            
            if not all_predictions:
                raise RuntimeError("All ensemble models failed to predict")
            
            # Combine predictions using the weights
            if self.task_type == 'classification':
                # For classification, use weighted voting
                # First, get all unique classes
                unique_classes = np.unique(np.concatenate([pred.reshape(-1) for pred in all_predictions]))
                
                # Initialize vote counts
                votes = np.zeros((X.shape[0], len(unique_classes)))
                
                # Count weighted votes for each class
                for i, preds in enumerate(all_predictions):
                    for j, pred in enumerate(preds):
                        class_idx = np.where(unique_classes == pred)[0][0]
                        votes[j, class_idx] += self.weights[i]
                
                # Return the class with the most votes
                return unique_classes[np.argmax(votes, axis=1)]
            else:
                # For regression, use weighted average
                all_predictions = np.array(all_predictions)
                weighted_sum = np.zeros(X.shape[0])
                
                for i, preds in enumerate(all_predictions):
                    weighted_sum += self.weights[i] * preds
                
                return weighted_sum
                
        elif self.hybrid_type == 'quantum_enhanced':
            # Extract quantum features
            quantum_features = self._quantum_feature_extraction(X)
            
            # Combine original features with quantum features
            X_enhanced = np.hstack([X, quantum_features]) if quantum_features is not None else X
            
            # Use the classical model to predict
            return self.classical_model.predict(X_enhanced)
        
        else:
            raise ValueError(f"Unsupported hybrid type: {self.hybrid_type}")
    
    def predict_proba(self, X):
        """
        Get probability estimates for classification.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Probability estimates
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        if self.task_type != 'classification':
            raise ValueError("Probability prediction only available for classification tasks")
        
        if self.hybrid_type == 'feature_hybrid':
            # Select features for classical and quantum models
            quantum_features, classical_features, _, _ = self._select_features(X)
            
            # Get probability estimates from quantum model
            try:
                quantum_proba = self.quantum_model.predict_proba(quantum_features) if quantum_features is not None else None
            except (ValueError, AttributeError):
                quantum_proba = None
            
            # Get probability estimates from classical model
            try:
                classical_proba = self.classical_model.predict_proba(classical_features) if classical_features is not None else None
            except (ValueError, AttributeError):
                classical_proba = None
            
            # Combine probability estimates
            if quantum_proba is not None and classical_proba is not None:
                # Ensure both have the same number of classes
                if quantum_proba.shape[1] != classical_proba.shape[1]:
                    # If different number of classes, use the one with more classes
                    if quantum_proba.shape[1] > classical_proba.shape[1]:
                        # Extend classical probabilities
                        extended_proba = np.zeros_like(quantum_proba)
                        extended_proba[:, :classical_proba.shape[1]] = classical_proba
                        classical_proba = extended_proba
                    else:
                        # Extend quantum probabilities
                        extended_proba = np.zeros_like(classical_proba)
                        extended_proba[:, :quantum_proba.shape[1]] = quantum_proba
                        quantum_proba = extended_proba
                
                # Weight the probabilities
                combined_proba = self.quantum_percentage * quantum_proba + \
                                (1 - self.quantum_percentage) * classical_proba
                
                # Normalize to ensure they sum to 1
                row_sums = combined_proba.sum(axis=1)
                combined_proba = combined_proba / row_sums[:, np.newaxis]
                
                return combined_proba
            elif quantum_proba is not None:
                return quantum_proba
            elif classical_proba is not None:
                return classical_proba
            else:
                raise RuntimeError("Both quantum and classical models failed to provide probability estimates")
                
        elif self.hybrid_type == 'ensemble':
            # Get probability estimates from all models
            all_probas = []
            for _, model in self.models:
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        all_probas.append(proba)
                except Exception as e:
                    print(f"Warning: Model probability estimation failed: {e}")
            
            if not all_probas:
                raise RuntimeError("All ensemble models failed to provide probability estimates")
            
            # Find the maximum number of classes
            max_classes = max(proba.shape[1] for proba in all_probas)
            
            # Extend all probability arrays to have the same number of classes
            extended_probas = []
            for proba in all_probas:
                if proba.shape[1] < max_classes:
                    extended = np.zeros((proba.shape[0], max_classes))
                    extended[:, :proba.shape[1]] = proba
                    extended_probas.append(extended)
                else:
                    extended_probas.append(proba)
            
            # Combine probability estimates using the weights
            combined_proba = np.zeros((X.shape[0], max_classes))
            for i, proba in enumerate(extended_probas):
                combined_proba += self.weights[i] * proba
            
            # Normalize to ensure they sum to 1
            row_sums = combined_proba.sum(axis=1)
            combined_proba = combined_proba / row_sums[:, np.newaxis]
            
            return combined_proba
                
        elif self.hybrid_type == 'quantum_enhanced':
            # Extract quantum features
            quantum_features = self._quantum_feature_extraction(X)
            
            # Combine original features with quantum features
            X_enhanced = np.hstack([X, quantum_features]) if quantum_features is not None else X
            
            # Use the classical model to predict probabilities
            if hasattr(self.classical_model, 'predict_proba'):
                return self.classical_model.predict_proba(X_enhanced)
            else:
                raise ValueError("Classical model does not support probability estimates")
        
        else:
            raise ValueError(f"Unsupported hybrid type: {self.hybrid_type}")
    
    def get_model_size(self):
        """
        Estimate the hybrid model size.
        
        Returns:
            int: Approximate model size in bytes
        """
        total_size = 0
        
        if self.hybrid_type == 'feature_hybrid':
            # Add size of classical model
            if self.classical_model is not None:
                total_size += self.classical_model.get_model_size()
            
            # Add size of quantum model
            if self.quantum_model is not None:
                total_size += self.quantum_model.get_model_size()
                
        elif self.hybrid_type == 'ensemble':
            # Add size of all models
            for _, model in self.models:
                if hasattr(model, 'get_model_size'):
                    total_size += model.get_model_size()
                
        elif self.hybrid_type == 'quantum_enhanced':
            # Add size of classical model
            if self.classical_model is not None:
                total_size += self.classical_model.get_model_size()
            
            # Add size of quantum model
            if self.quantum_model is not None:
                total_size += self.quantum_model.get_model_size()
            
            # Add size of PCA
            if self.pca is not None:
                import sys
                import pickle
                try:
                    total_size += sys.getsizeof(pickle.dumps(self.pca))
                except:
                    pass
        
        return total_size
    
    def benchmark_inference_time(self, X, num_samples=100):
        """
        Measure inference time.
        
        Args:
            X (array-like): Features to predict
            num_samples (int): Number of samples to use for averaging
            
        Returns:
            float: Average inference time per sample in milliseconds
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before benchmarking")
        
        # Use a subset of data if X is large
        if len(X) > num_samples:
            indices = np.random.choice(len(X), num_samples, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
            
        # Warmup
        self.predict(X_subset[:5])
        
        # Measure time
        start_time = time.time()
        self.predict(X_subset)
        end_time = time.time()
        
        # Calculate average time per sample in milliseconds
        avg_time_ms = (end_time - start_time) * 1000 / len(X_subset)
        
        return avg_time_ms


def get_hybrid_model_scores(hybrid_configs, X_train, X_test, y_train, y_test, task_type='classification'):
    """
    Train and evaluate multiple hybrid models.
    
    Args:
        hybrid_configs (list): List of hybrid configurations to evaluate, each a dict with keys:
                              'hybrid_type', 'classical_model_type', 'quantum_framework', etc.
        X_train, X_test (array-like): Training and test features
        y_train, y_test (array-like): Training and test targets
        task_type (str): Type of task ('classification' or 'regression')
        
    Returns:
        dict: Dictionary with model names and performance metrics
    """
    results = {}
    
    for config in hybrid_configs:
        hybrid_type = config.get('hybrid_type', 'feature_hybrid')
        classical_model_type = config.get('classical_model_type', 'random_forest')
        quantum_framework = config.get('quantum_framework', 'qiskit')
        quantum_model_type = config.get('quantum_model_type', 'variational')
        n_qubits = config.get('n_qubits', None)
        quantum_percentage = config.get('quantum_percentage', 0.3)
        
        model_name = f"Hybrid_{hybrid_type}_{quantum_framework}"
        print(f"\nTraining {model_name}...")
        
        model = HybridModel(
            hybrid_type=hybrid_type,
            classical_model_type=classical_model_type,
            quantum_framework=quantum_framework,
            quantum_model_type=quantum_model_type,
            task_type=task_type,
            n_qubits=n_qubits,
            quantum_percentage=quantum_percentage
        )
        
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions for classification
        y_proba = None
        if task_type == 'classification':
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Measure inference time
        inference_time = model.benchmark_inference_time(X_test)
        
        # Get model size
        model_size = model.get_model_size()
        
        # Store results
        results[model_name] = {
            'model': model,
            'training_time': training_time,
            'inference_time': inference_time,
            'model_size': model_size,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Inference time: {inference_time:.2f} ms per sample")
        
    return results 