import os
from pathlib import Path
import sys

class MultipleYamlError(Exception):
    """Custom exception for when multiple YAML files are found in a directory"""
    pass

def find_yaml_files(base_path, filter_for_keyword):
    """
    Find YAML files in immediate subdirectories of the given path.
    Throws an error if multiple YAML files are found in any subdirectory.
    
    Args:
        base_path (str): The base directory path to search in
    
    Raises:
        MultipleYamlError: If multiple YAML files are found in any subdirectory
    """
    base = Path(base_path)
    
    if not base.exists():
        print(f"Error: Path '{base_path}' does not exist")
        return
    if not base.is_dir():
        print(f"Error: '{base_path}' is not a directory")
        return
    
    for item in base.iterdir():
        if item.is_dir():
            yaml_files = []
            # Collect all yaml files in the subdirectory
            for file in item.iterdir():
                if file.is_file() and file.suffix.lower() in ['.yaml', '.yml']:
                    yaml_files.append(file)
            
            if len(yaml_files) > 1:
                yaml_paths = [str(f) for f in yaml_files]
                raise MultipleYamlError(
                    f"Multiple YAML files found in directory '{item}': {yaml_paths}"
                )
            elif len(yaml_files) == 1:
                if filter_for_keyword in str(yaml_files[0]):
                    print(f"CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config {yaml_files[0]}")
            else:
                print(f"No YAML file found in directory: {item.name}")

if __name__ == "__main__":
    try:
        path = sys.argv[1]
        try:
            filter_for_keyword = sys.argv[2]
        except:
            filter_for_keyword = ""
        find_yaml_files(path, filter_for_keyword)
    except MultipleYamlError as e:
        print(f"Error: {e}")