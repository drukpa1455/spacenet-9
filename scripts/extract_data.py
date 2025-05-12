#!/usr/bin/env python
"""
Extract SpaceNet 9 challenge data from ZIP files.
"""

import os
import argparse
import zipfile
import yaml
import shutil
from pathlib import Path
from glob import glob
from typing import List, Dict, Optional, Any
from tqdm import tqdm


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> bool:
    """Extract a ZIP file to a directory.
    
    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract to
        force: If True, extract even if directory already has content
        
    Returns:
        True if extraction was successful, False otherwise
    """
    # Check if extraction directory already has content
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        print(f"Directory {extract_dir} already has content. Skipping extraction. Use --force to re-extract.")
        return True
    
    print(f"Extracting {zip_path} to {extract_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all files in the archive
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            total_size = sum(zip_info.file_size for zip_info in zip_ref.infolist())
            
            # If directory exists and force is True, clean it first
            if extract_dir.exists() and force:
                # Instead of removing the directory which could cause issues,
                # we'll extract to a temporary directory and then move files
                temp_dir = extract_dir.with_name(f"{extract_dir.name}_temp")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                extract_target = temp_dir
            else:
                extract_target = extract_dir
                
            # Extract with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Extracting {zip_path.name}") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_target)
                    file_info = zip_ref.getinfo(file)
                    pbar.update(file_info.file_size)
            
            # If we used a temp directory, move files to the target directory
            if extract_target != extract_dir:
                # Move content from temp directory to target directory
                for item in temp_dir.iterdir():
                    shutil.move(str(item), str(extract_dir))
                # Remove temp directory
                shutil.rmtree(temp_dir)
            
            print(f"Successfully extracted {total_files} files to {extract_dir}")
            return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def main() -> None:
    """Main function to extract SpaceNet 9 data."""
    parser = argparse.ArgumentParser(description="Extract SpaceNet 9 challenge data")
    parser.add_argument("--config", type=str, default="configs/data_urls.yaml", 
                        help="Path to configuration file with paths")
    parser.add_argument("--datasets", nargs="+", 
                        choices=["train", "public_test", "baseline_submission", 
                                "trivial_submission", "perfect_solution", "scorer", "all"],
                        default=["all"], help="Datasets to extract")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-extraction even if directories already have content")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    raw_data_dir = Path(config["paths"]["raw"])
    processed_data_dir = Path(config["paths"]["processed"])
    
    # Find all ZIP files in raw data directory
    zip_files = glob(str(raw_data_dir / "*.zip"))
    
    if not zip_files:
        print(f"No ZIP files found in {raw_data_dir}")
        return
    
    # Determine which datasets to extract
    if "all" in args.datasets:
        # Extract all ZIP files
        for zip_path in zip_files:
            filename = os.path.basename(zip_path)
            dataset_name = filename.split(".")[0]
            extract_dir = processed_data_dir / dataset_name
            extract_zip(Path(zip_path), extract_dir, force=args.force)
    else:
        # Extract only specified datasets
        for dataset in args.datasets:
            # Look for the corresponding ZIP file
            dataset_file = None
            for zip_path in zip_files:
                if dataset in os.path.basename(zip_path).lower():
                    dataset_file = zip_path
                    break
            
            if dataset_file:
                extract_dir = processed_data_dir / dataset
                extract_zip(Path(dataset_file), extract_dir, force=args.force)
            else:
                print(f"Warning: ZIP file for {dataset} not found")
    
    print("Extraction process completed")


if __name__ == "__main__":
    main() 