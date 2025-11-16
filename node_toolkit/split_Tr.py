import os
import shutil
import json

def organize_data_by_split(split, input_dir, splits_final_path):
    """
    Organize data into train and val folders based on the specified split from splits_final.json.

    Args:
        split (int): The fold number to use for organizing data (1-based index).
        input_dir (str): The directory containing the original data files.
        splits_final_path (str): The path to the splits_final.json file.
    """
    # Load splits_final.json
    with open(splits_final_path, 'r') as f:
        splits = json.load(f)
    
    # Ensure the specified split is valid
    if split < 1 or split > len(splits):
        raise ValueError(f"Invalid split number. Please choose a number between 1 and {len(splits)}")
    
    # Get train and val cases for the specified split
    train_cases = splits[split - 1]['train']
    val_cases = splits[split - 1]['val']
    
    # Create train and val directories if they don't exist
    train_dir = os.path.join(input_dir, 'train')
    val_dir = os.path.join(input_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Move files to train directory
    for case in train_cases:
        case_files = [f for f in os.listdir(input_dir) if f.startswith(case)]
        for file in case_files:
            shutil.move(os.path.join(input_dir, file), os.path.join(train_dir, file))
    
    # Move files to val directory
    for case in val_cases:
        case_files = [f for f in os.listdir(input_dir) if f.startswith(case)]
        for file in case_files:
            shutil.move(os.path.join(input_dir, file), os.path.join(val_dir, file))

    print(f"Data organized for split {split}. Train files moved to '{train_dir}' and validation files moved to '{val_dir}'.")

# Example usage
if __name__ == "__main__":
    input_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Data\TrainNiigzCsvData\Tr"
    splits_final_path = r"C:\Users\PC\PycharmProjects\thu_xwh\nnUNet\nnUNet_raw500\Dataset500_plaque\splits_final.json"
    split = 1  # Use the first fold for organizing data

    organize_data_by_split(split, input_dir, splits_final_path)
