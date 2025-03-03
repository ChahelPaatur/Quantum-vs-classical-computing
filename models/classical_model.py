"""
Classical machine learning model implementations for benchmarking.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


class ClassicalModel:
    """Base class for classical machine learning models."""
    
    def __init__(self, model_type='random_forest', task_type='classification'):
        """
        Initialize a classical model.
        
        Args:
            model_type (str): Type of model ('random_forest', 'gradient_boosting', 'neural_network', 'svm', 'deep_learning')
            task_type (str): Type of task ('classification' or 'regression')
        """
        self.model_type = model_type
        self.task_type = task_type
        self.model = None
        self.is_fitted = False
        self.is_deep_learning = model_type == 'deep_learning'
    
    def _create_model(self):
        """
        Create the appropriate classical model based on model_type and task_type.
        """
        if self.model_type == 'random_forest':
            if self.task_type == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                )
            else:  # regression
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                )
                
        elif self.model_type == 'gradient_boosting':
            if self.task_type == 'classification':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            else:  # regression
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                
        elif self.model_type == 'neural_network':
            if self.task_type == 'classification':
                self.model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    max_iter=300,
                    random_state=42
                )
            else:  # regression
                self.model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    max_iter=300,
                    random_state=42
                )
                
        elif self.model_type == 'svm':
            if self.task_type == 'classification':
                self.model = SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            else:  # regression
                self.model = SVR(
                    C=1.0,
                    kernel='rbf'
                )
                
        elif self.model_type == 'deep_learning':
            # For deep learning, we'll create the model when we know the input shape
            self.model = None
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_deep_learning_model(self, input_shape, num_classes=None):
        """
        Create a deep learning model using TensorFlow/Keras.
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int, optional): Number of classes for classification
        """
        model = Sequential()
        
        # First layer with input shape
        model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
        model.add(Dropout(0.2))
        
        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        
        # Output layer
        if self.task_type == 'classification':
            if num_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:  # regression
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X, y):
        """
        Train the classical model.
        
        Args:
            X (array-like): Training features
            y (array-like): Target values
            
        Returns:
            ClassicalModel: Trained model instance
        """
        if self.model is None:
            self._create_model()
        
        if self.is_deep_learning:
            # Convert data to right format for deep learning
            X = np.array(X)
            
            # For classification, convert labels to categorical
            if self.task_type == 'classification':
                classes = np.unique(y)
                num_classes = len(classes)
                
                # One-hot encode if more than 2 classes
                if num_classes > 2:
                    y_encoded = to_categorical(pd.Categorical(y).codes)
                else:
                    y_encoded = y
            else:
                y_encoded = y
                num_classes = None
            
            # Create the deep learning model
            self.model = self._create_deep_learning_model(X.shape[1], num_classes)
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train the model
            self.model.fit(
                X, y_encoded,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            # Train sklearn model
            self.model.fit(X, y)
        
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
        
        if self.is_deep_learning:
            X = np.array(X)
            if self.task_type == 'classification':
                raw_preds = self.model.predict(X)
                
                # For binary classification
                if raw_preds.shape[1] == 1:
                    return (raw_preds > 0.5).astype(int).flatten()
                
                # For multi-class
                return np.argmax(raw_preds, axis=1)
            else:
                return self.model.predict(X).flatten()
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates.
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Probability estimates
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        if self.task_type != 'classification':
            raise ValueError("Probability prediction only available for classification tasks")
        
        if self.is_deep_learning:
            X = np.array(X)
            return self.model.predict(X)
        else:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise ValueError(f"Model {self.model_type} does not support probability predictions")
    
    def tune_hyperparameters(self, X, y, param_grid=None):
        """
        Tune hyperparameters using grid search.
        
        Args:
            X (array-like): Training features
            y (array-like): Target values
            param_grid (dict, optional): Dictionary of parameters to tune
            
        Returns:
            ClassicalModel: Model with tuned hyperparameters
        """
        if self.is_deep_learning:
            print("Hyperparameter tuning not implemented for deep learning models")
            return self
            
        if self.model is None:
            self._create_model()
            
        if param_grid is None:
            # Default parameter grids for different model types
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5]
                }
            elif self.model_type == 'neural_network':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                }
            else:
                param_grid = {}
                
        # Set up scoring for different task types
        scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        # Create and fit the grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return self
    
    def get_model_size(self):
        """
        Estimate the model size in memory.
        
        Returns:
            int: Approximate model size in bytes
        """
        import sys
        
        if self.model is None:
            return 0
            
        if self.is_deep_learning:
            # For deep learning models, estimate the number of parameters
            return sum(np.prod(w.shape) for w in self.model.get_weights()) * 4  # 4 bytes per float32
        else:
            # For sklearn models, use pickle size estimation
            import pickle
            return sys.getsizeof(pickle.dumps(self.model))
    
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
        if self.is_deep_learning:
            self.model.predict(np.array(X_subset[:5]))
        else:
            self.model.predict(X_subset[:5])
            
        # Measure time
        start_time = time.time()
        
        if self.is_deep_learning:
            self.model.predict(np.array(X_subset))
        else:
            self.model.predict(X_subset)
            
        end_time = time.time()
        
        # Calculate average time per sample in milliseconds
        avg_time_ms = (end_time - start_time) * 1000 / len(X_subset)
        
        return avg_time_ms


def get_classical_model_scores(model_types, X_train, X_test, y_train, y_test, task_type='classification'):
    """
    Train and evaluate multiple classical models.
    
    Args:
        model_types (list): List of model types to evaluate
        X_train, X_test (array-like): Training and test features
        y_train, y_test (array-like): Training and test targets
        task_type (str): Type of task ('classification' or 'regression')
        
    Returns:
        dict: Dictionary with model names and performance metrics
    """
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        model = ClassicalModel(model_type=model_type, task_type=task_type)
        
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
        results[f"Classical_{model_type}"] = {
            'model': model,
            'training_time': training_time,
            'inference_time': inference_time,
            'model_size': model_size,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Inference time: {inference_time:.2f} ms per sample")
        print(f"  Model size: {model_size / (1024*1024):.2f} MB")
    
    return results 