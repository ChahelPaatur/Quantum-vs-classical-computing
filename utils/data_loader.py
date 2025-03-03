"""
Utilities for downloading and processing Kaggle datasets.
"""
import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataLoader:
    def __init__(self, dataset_path='data'):
        """
        Initialize the DataLoader with the path to store datasets.
        
        Args:
            dataset_path (str): Path to store downloaded datasets
        """
        self.dataset_path = dataset_path
        os.makedirs(dataset_path, exist_ok=True)
        self.api = None
        
    def authenticate_kaggle(self):
        """Authenticate with Kaggle API"""
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            print("Successfully authenticated with Kaggle API")
            return True
        except Exception as e:
            print(f"Kaggle authentication failed: {e}")
            print("Please ensure you have a kaggle.json file with your API credentials")
            return False
    
    def download_dataset(self, dataset_name):
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_name (str): Name of the dataset in format 'owner/dataset-name'
        
        Returns:
            str: Path to the downloaded dataset
        """
        if self.api is None and not self.authenticate_kaggle():
            return None
            
        try:
            print(f"Downloading dataset: {dataset_name}")
            self.api.dataset_download_files(
                dataset_name, 
                path=self.dataset_path, 
                unzip=True
            )
            # Get the folder name where dataset was downloaded
            dataset_folder = os.path.join(self.dataset_path, dataset_name.split('/')[-1])
            return dataset_folder
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return None
    
    def load_dataset(self, file_path, task_type='classification'):
        """
        Load a dataset from a file path.
        
        Args:
            file_path (str): Path to the dataset file
            task_type (str): Type of task ('classification' or 'regression')
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, preprocessor
        """
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Print dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Dataset columns: {data.columns.tolist()}")
        
        # Basic dataset analysis
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        print(f"Numeric columns: {len(numeric_columns)}")
        print(f"Categorical columns: {len(categorical_columns)}")
        
        # Ask user to select target column if not specified
        if 'target' not in data.columns:
            print("\nAvailable columns for target:")
            for i, col in enumerate(data.columns):
                print(f"{i}: {col}")
            target_col = input("Enter the column number to use as target: ")
            target_col = data.columns[int(target_col)]
        else:
            target_col = 'target'
        
        # Split features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
        )
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )
        
        # Fit and transform the data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        print(f"Processed training data shape: {X_train_processed.shape}")
        print(f"Processed test data shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    
    def get_dataset_suggestions(self, task_type='classification'):
        """
        Get suggestions for popular datasets based on task type.
        
        Args:
            task_type (str): Type of task ('classification' or 'regression')
            
        Returns:
            list: List of suggested dataset names
        """
        if self.api is None and not self.authenticate_kaggle():
            return []
        
        suggestions = []
        if task_type == 'classification':
            suggestions = [
                'uciml/iris',
                'uciml/breast-cancer-wisconsin-data',
                'uciml/adult-census-income',
                'uciml/wine-quality-dataset',
                'uciml/heart-disease-dataset'
            ]
        elif task_type == 'regression':
            suggestions = [
                'uciml/boston-housing-dataset',
                'shree1992/houserent-dataset',
                'uciml/auto-mpg-dataset',
                'uciml/power-plant-dataset',
                'uciml/bike-sharing-dataset'
            ]
        
        return suggestions


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    if loader.authenticate_kaggle():
        print("Suggested classification datasets:")
        for ds in loader.get_dataset_suggestions('classification'):
            print(f"- {ds}") 