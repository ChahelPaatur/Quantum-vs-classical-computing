"""
Quantum machine learning model implementations for benchmarking.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier

# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC, QSVC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit.primitives import Sampler

# Pennylane imports
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers


class QuantumModel:
    """Base class for quantum machine learning models."""
    
    def __init__(self, framework='qiskit', model_type='variational', task_type='classification', n_qubits=None):
        """
        Initialize a quantum model.
        
        Args:
            framework (str): Quantum framework to use ('qiskit' or 'pennylane')
            model_type (str): Type of quantum model ('variational', 'kernel', 'circuit')
            task_type (str): Type of task ('classification' or 'regression')
            n_qubits (int, optional): Number of qubits to use (default: auto-determined from data)
        """
        self.framework = framework
        self.model_type = model_type
        self.task_type = task_type
        self.n_qubits = n_qubits
        self.model = None
        self.is_fitted = False
        self.feature_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
        self.num_classes = None
        
        # For PennyLane models
        self.device = None
        self.qnode = None
        self.weight_shapes = None
        self.trained_weights = None
        
    def _auto_determine_qubits(self, X):
        """
        Automatically determine the number of qubits based on feature dimensions.
        
        Args:
            X (array-like): Input features
            
        Returns:
            int: Number of qubits to use
        """
        n_features = X.shape[1]
        n_qubits = max(int(np.ceil(np.log2(n_features))), 2)
        return min(n_qubits, 10)  # Cap at 10 qubits for practicality
    
    def _prepare_data(self, X):
        """
        Scale and prepare data for quantum processing.
        
        Args:
            X (array-like): Input features
            
        Returns:
            array: Scaled features
        """
        # Reduce dimensionality if needed
        if hasattr(self, 'n_qubits') and self.n_qubits is not None:
            if X.shape[1] > self.n_qubits:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.n_qubits)
                X = pca.fit_transform(X)
        
        # Scale features to [0, 2π] for angle encoding
        return self.feature_scaler.fit_transform(X)
    
    def _create_qiskit_model(self, n_features, num_classes=None):
        """
        Create a Qiskit quantum model.
        
        Args:
            n_features (int): Number of input features
            num_classes (int, optional): Number of classes for classification
        """
        if self.n_qubits is None:
            self.n_qubits = max(int(np.ceil(np.log2(n_features))), 2)
            
        # For practicality, limit the number of qubits
        self.n_qubits = min(self.n_qubits, 10)
        
        # Create appropriate quantum model based on model_type
        if self.model_type == 'variational':
            # Create feature map for data encoding
            feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)
            
            # Create variational form for trainable circuit
            ansatz = RealAmplitudes(self.n_qubits, reps=2)
            
            # Create the VQC model
            if self.task_type == 'classification':
                self.model = VQC(
                    feature_map=feature_map,
                    ansatz=ansatz,
                    optimizer=ADAM(maxiter=100),
                    callback=None
                )
            else:  # regression
                # For regression, we'll still use VQC but interpret output differently
                self.model = VQC(
                    feature_map=feature_map,
                    ansatz=ansatz,
                    optimizer=ADAM(maxiter=100),
                    callback=None
                )
                
        elif self.model_type == 'kernel':
            # Use Quantum Support Vector Classifier
            if self.task_type == 'classification':
                feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)
                self.model = QSVC(
                    quantum_kernel=qiskit.algorithms.kernels.QuantumKernel(
                        feature_map=feature_map,
                        quantum_instance=Aer.get_backend('qasm_simulator')
                    )
                )
                
                # For multi-class, wrap with OneVsRestClassifier
                if num_classes is not None and num_classes > 2:
                    self.model = OneVsRestClassifier(self.model)
            else:
                raise ValueError("Kernel method not implemented for regression tasks")
                
        elif self.model_type == 'circuit':
            # Build a quantum neural network using TwoLayerQNN
            feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)
            ansatz = RealAmplitudes(self.n_qubits, reps=2)
            
            # Combine feature map and ansatz
            circuit = QuantumCircuit(self.n_qubits)
            circuit.compose(feature_map, inplace=True)
            circuit.compose(ansatz, inplace=True)
            
            if self.task_type == 'classification':
                # Create a QNN classifier
                sampler = Sampler()
                qnn = TwoLayerQNN(
                    num_inputs=n_features,
                    num_qubits=self.n_qubits,
                    feature_map=feature_map,
                    ansatz=ansatz,
                    sampler=sampler
                )
                self.model = NeuralNetworkClassifier(
                    neural_network=qnn,
                    optimizer=SPSA(maxiter=100),
                    callback=None
                )
            else:
                raise ValueError("Circuit method not implemented for regression tasks")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_pennylane_model(self, n_features, num_classes=None):
        """
        Create a PennyLane quantum model.
        
        Args:
            n_features (int): Number of input features
            num_classes (int, optional): Number of classes for classification
        """
        if self.n_qubits is None:
            self.n_qubits = max(int(np.ceil(np.log2(n_features))), 2)
            
        # For practicality, limit the number of qubits
        self.n_qubits = min(self.n_qubits, 10)
        
        # Create a quantum device
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        
        # Define weights for the model
        if self.model_type == 'variational':
            # Define QNode structure with AngleEmbedding and StronglyEntanglingLayers
            @qml.qnode(self.device)
            def circuit(inputs, weights):
                # Embed the inputs into the quantum circuit
                AngleEmbedding(inputs, wires=range(self.n_qubits))
                
                # Variational circuit
                StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                
                # Measure the circuit
                if self.task_type == 'classification':
                    if num_classes is not None and num_classes > 2:
                        # Multi-class: measure multiple qubits
                        return [qml.expval(qml.PauliZ(i)) for i in range(min(num_classes, self.n_qubits))]
                    else:
                        # Binary: measure one qubit
                        return qml.expval(qml.PauliZ(0))
                else:  # regression
                    return qml.expval(qml.PauliZ(0))
            
            # Define parameter shapes
            self.weight_shapes = {"weights": (3, self.n_qubits, 3)}  # 3 layers of strongly entangling layers
            self.qnode = circuit
        
        elif self.model_type == 'kernel':
            # Define a kernel-based circuit
            @qml.qnode(self.device)
            def kernel_circuit(x1, x2):
                # Embed inputs
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    
                # Encode first input
                for i, x in enumerate(x1[:self.n_qubits]):
                    qml.RZ(x, wires=i)
                
                # Apply inverse operations for the second input
                for i, x in enumerate(x2[:self.n_qubits]):
                    qml.RZ(-x, wires=i)
                    
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                
                # Return probability of all zeros
                return qml.probs(wires=range(self.n_qubits))[0]
            
            self.qnode = kernel_circuit
            
        elif self.model_type == 'circuit':
            # Define a more complex circuit for direct learning
            @qml.qnode(self.device)
            def complex_circuit(inputs, weights_1, weights_2):
                # Embed inputs
                AngleEmbedding(inputs, wires=range(self.n_qubits))
                
                # First parameterized layer
                for i in range(self.n_qubits):
                    qml.Rot(*weights_1[i], wires=i)
                
                # Entangling layer
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
                # Second parameterized layer
                for i in range(self.n_qubits):
                    qml.Rot(*weights_2[i], wires=i)
                
                # Measure
                if self.task_type == 'classification':
                    if num_classes is not None and num_classes > 2:
                        return [qml.expval(qml.PauliZ(i)) for i in range(min(num_classes, self.n_qubits))]
                    else:
                        return qml.expval(qml.PauliZ(0))
                else:
                    return qml.expval(qml.PauliZ(0))
            
            self.weight_shapes = {
                "weights_1": (self.n_qubits, 3),  # 3 parameters per Rot gate
                "weights_2": (self.n_qubits, 3)   # 3 parameters per Rot gate
            }
            self.qnode = complex_circuit
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Train the quantum model.
        
        Args:
            X (array-like): Training features
            y (array-like): Target values
            
        Returns:
            QuantumModel: Trained model instance
        """
        # Process and scale data
        X_scaled = self._prepare_data(X)
        
        # Get unique classes for classification
        if self.task_type == 'classification':
            self.classes_ = np.unique(y)
            self.num_classes = len(self.classes_)
        
        # Create the model if not already created
        if self.framework == 'qiskit':
            if self.model is None:
                self._create_qiskit_model(X_scaled.shape[1], self.num_classes)
            
            # Fit the model
            self.model.fit(X_scaled, y)
            
        elif self.framework == 'pennylane':
            if self.qnode is None:
                self._create_pennylane_model(X_scaled.shape[1], self.num_classes)
            
            # Initialize weights
            if hasattr(self, 'weight_shapes') and self.weight_shapes is not None:
                np.random.seed(42)
                init_weights = {k: np.random.uniform(low=-np.pi, high=np.pi, size=s) 
                               for k, s in self.weight_shapes.items()}
                
                # Define cost function
                def cost(weights):
                    predictions = np.array([self.qnode(x, **{k: weights[k] for k in self.weight_shapes.keys()})
                                          for x in X_scaled])
                    
                    if self.task_type == 'classification':
                        if self.num_classes > 2:
                            # For multi-class, interpret outputs as per-class scores
                            if isinstance(predictions[0], list):
                                predictions = np.array(predictions)
                                # Convert scores to probabilities with softmax
                                predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
                                true_classes = pd.get_dummies(y).values
                                # Calculate cross entropy loss
                                return -np.mean(np.sum(true_classes * np.log(predictions + 1e-10), axis=1))
                            else:
                                # If not getting array output, fallback to binary classification
                                return np.mean((predictions - y) ** 2)
                        else:
                            # Binary classification
                            binary_preds = (predictions + 1) / 2  # Convert from [-1, 1] to [0, 1]
                            return np.mean((binary_preds - y) ** 2)
                    else:
                        # Regression
                        return np.mean((predictions - y) ** 2)
                
                # Optimize the weights
                opt = qml.AdamOptimizer(stepsize=0.1)
                self.trained_weights = init_weights.copy()
                
                # Run optimization for a number of steps
                for _ in range(100):
                    self.trained_weights = opt.step(cost, self.trained_weights)
            
            elif self.model_type == 'kernel':
                # For kernel methods, we store the training data
                self.X_train = X_scaled
                self.y_train = y
                
                # For kernel-based learning, we'll use a simple SVM approach
                from sklearn.svm import SVC
                
                # Create the kernel matrix
                n_samples = len(X_scaled)
                kernel_matrix = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    for j in range(i, n_samples):
                        # Compute kernel value between samples
                        kernel_val = self.qnode(X_scaled[i], X_scaled[j])
                        kernel_matrix[i, j] = kernel_val
                        kernel_matrix[j, i] = kernel_val
                
                # Use SVC with precomputed kernel
                self.model = SVC(kernel='precomputed')
                self.model.fit(kernel_matrix, y)
                
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
            
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X (array-like): Features to predict
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Scale the input data
        X_scaled = self.feature_scaler.transform(X)
        
        if self.framework == 'qiskit':
            return self.model.predict(X_scaled)
            
        elif self.framework == 'pennylane':
            if hasattr(self, 'trained_weights') and self.trained_weights is not None:
                # Make predictions using the trained weights
                predictions = np.array([self.qnode(x, **{k: self.trained_weights[k] for k in self.weight_shapes.keys()})
                                      for x in X_scaled])
                
                if self.task_type == 'classification':
                    if self.num_classes > 2 and isinstance(predictions[0], list):
                        # Convert to class indices for multi-class
                        predictions = np.array(predictions)
                        return np.argmax(predictions, axis=1)
                    else:
                        # Binary classification
                        return (predictions > 0).astype(int)
                else:
                    # Regression
                    return predictions
                    
            elif self.model_type == 'kernel':
                # For kernel-based prediction
                n_train = len(self.X_train)
                n_test = len(X_scaled)
                kernel_matrix = np.zeros((n_test, n_train))
                
                for i in range(n_test):
                    for j in range(n_train):
                        kernel_matrix[i, j] = self.qnode(X_scaled[i], self.X_train[j])
                
                return self.model.predict(kernel_matrix)
            
            else:
                raise RuntimeError("Model not properly trained")
        
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
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
            
        # Scale the input data
        X_scaled = self.feature_scaler.transform(X)
        
        if self.framework == 'qiskit':
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                raise ValueError("This quantum model does not support probability estimates")
                
        elif self.framework == 'pennylane':
            if hasattr(self, 'trained_weights') and self.trained_weights is not None:
                # Get raw outputs
                predictions = np.array([self.qnode(x, **{k: self.trained_weights[k] for k in self.weight_shapes.keys()})
                                      for x in X_scaled])
                
                if self.num_classes > 2 and isinstance(predictions[0], list):
                    # Multi-class: convert quantum outputs to probabilities
                    predictions = np.array(predictions)
                    # Apply softmax
                    probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
                    return probs
                else:
                    # Binary classification: convert from [-1, 1] to [0, 1]
                    probs = (predictions + 1) / 2
                    # Format as binary probabilities [p(0), p(1)]
                    return np.column_stack([1 - probs, probs])
                    
            elif self.model_type == 'kernel' and hasattr(self.model, 'predict_proba'):
                # For kernel-based prediction
                n_train = len(self.X_train)
                n_test = len(X_scaled)
                kernel_matrix = np.zeros((n_test, n_train))
                
                for i in range(n_test):
                    for j in range(n_train):
                        kernel_matrix[i, j] = self.qnode(X_scaled[i], self.X_train[j])
                
                return self.model.predict_proba(kernel_matrix)
            
            else:
                raise ValueError("This quantum model does not support probability estimates")
        
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def get_model_size(self):
        """
        Estimate the quantum model size.
        
        Returns:
            int: Approximate model size in bytes
        """
        # For quantum models, we estimate size based on the number of qubits and circuit depth
        if self.framework == 'qiskit':
            # Estimate based on qiskit model parameters
            if self.model is None:
                return 0
                
            import sys
            import pickle
            
            try:
                # Try to directly estimate the model size
                return sys.getsizeof(pickle.dumps(self.model))
            except:
                # If that fails, provide an estimate based on qubits
                # Quantum circuits grow exponentially with qubits
                # Each qubit doubles the state space
                return self.n_qubits * 2**self.n_qubits * 16  # 16 bytes per complex amplitude
                
        elif self.framework == 'pennylane':
            # Estimate based on the size of the trained weights
            if hasattr(self, 'trained_weights') and self.trained_weights is not None:
                total_params = sum(np.prod(s.shape) for s in self.trained_weights.values())
                return total_params * 8  # 8 bytes per parameter (float64)
            elif self.model_type == 'kernel':
                # For kernel methods, size depends on the training data
                return len(self.X_train) * len(self.X_train[0]) * 8
            else:
                # Fallback to qubit-based estimation
                return self.n_qubits * 2**self.n_qubits * 16
        
        else:
            return 0
    
    def benchmark_inference_time(self, X, num_samples=10):
        """
        Measure quantum inference time.
        
        Args:
            X (array-like): Features to predict
            num_samples (int): Number of samples to use for averaging
            
        Returns:
            float: Average inference time per sample in milliseconds
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before benchmarking")
        
        # Use fewer samples for quantum models as they're slower
        if len(X) > num_samples:
            indices = np.random.choice(len(X), num_samples, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        # Scale the data
        X_scaled = self.feature_scaler.transform(X_subset)
        
        # Warmup
        if len(X_scaled) > 0:
            if self.framework == 'qiskit':
                self.model.predict(X_scaled[:1])
            elif self.framework == 'pennylane':
                if hasattr(self, 'trained_weights') and self.trained_weights is not None:
                    self.qnode(X_scaled[0], **{k: self.trained_weights[k] for k in self.weight_shapes.keys()})
                elif self.model_type == 'kernel':
                    self.predict(X_scaled[:1])
        
        # Measure time
        start_time = time.time()
        
        # Make predictions
        self.predict(X_scaled)
        
        end_time = time.time()
        
        # Calculate average time per sample in milliseconds
        avg_time_ms = (end_time - start_time) * 1000 / len(X_scaled)
        
        return avg_time_ms


def get_quantum_model_scores(model_configs, X_train, X_test, y_train, y_test, task_type='classification'):
    """
    Train and evaluate multiple quantum models.
    
    Args:
        model_configs (list): List of model configurations to evaluate, each a dict with keys:
                              'framework', 'model_type', etc.
        X_train, X_test (array-like): Training and test features
        y_train, y_test (array-like): Training and test targets
        task_type (str): Type of task ('classification' or 'regression')
        
    Returns:
        dict: Dictionary with model names and performance metrics
    """
    results = {}
    
    for config in model_configs:
        framework = config.get('framework', 'qiskit')
        model_type = config.get('model_type', 'variational')
        n_qubits = config.get('n_qubits', None)
        
        model_name = f"Quantum_{framework}_{model_type}"
        print(f"\nTraining {model_name}...")
        
        model = QuantumModel(
            framework=framework,
            model_type=model_type,
            task_type=task_type,
            n_qubits=n_qubits
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
        print(f"  Qubits used: {model.n_qubits}")
        
    return results 