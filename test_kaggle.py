from utils.data_loader import DataLoader

def test_kaggle_api():
    # Create a data loader instance
    loader = DataLoader()
    
    # Authenticate with Kaggle
    success = loader.authenticate_kaggle()
    if not success:
        print("Authentication failed. Make sure your kaggle.json file is in the right location.")
        return
    
    # Get dataset suggestions
    print("Getting dataset suggestions...")
    suggestions = loader.get_dataset_suggestions('classification')
    print(f"Found {len(suggestions)} suggested datasets:")
    for idx, dataset in enumerate(suggestions[:5], 1):
        print(f"{idx}. {dataset}")
    

if __name__ == "__main__":
    test_kaggle_api() 