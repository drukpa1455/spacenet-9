#!/usr/bin/env python
"""
Explore the SpaceNet 9 dataset structure and visualize samples.
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple


def explore_directory_structure(base_dir: Path) -> None:
    """Explore and print the directory structure of the dataset using os.walk.
    
    Args:
        base_dir: Path to the dataset directory
    """
    print(f"\nDirectory structure in {base_dir}:")
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(str(base_dir), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        if level <= 2:  # Only show files at the first 3 levels to avoid clutter
            file_indent = ' ' * 4 * (level + 1)
            for file in files[:5]:  # Show only the first 5 files
                file_path = os.path.join(root, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb > 1:
                    print(f"{file_indent}{file} ({size_mb:.1f} MB)")
                else:
                    print(f"{file_indent}{file}")
            
            if len(files) > 5:
                print(f"{file_indent}... ({len(files) - 5} more files)")


def list_dataset_files(dataset_dir: Path) -> None:
    """List important files in the dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    print(f"\nFiles in {dataset_dir}:")
    
    # Check if dataset directory exists inside processed directory
    if not dataset_dir.exists():
        print(f"Directory {dataset_dir} does not exist")
        return
        
    nested_dir = dataset_dir / dataset_dir.name
    if nested_dir.exists():
        dataset_dir = nested_dir
    
    # List all TIF and CSV files
    tif_files = list(dataset_dir.glob("*.tif"))
    csv_files = list(dataset_dir.glob("*.csv"))
    
    if tif_files:
        print("\nTIF files:")
        for file in tif_files:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.1f} MB)")
    else:
        print("No TIF files found")
        
    if csv_files:
        print("\nCSV files:")
        for file in csv_files:
            with open(file, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  {file.name} ({line_count} rows)")
    else:
        print("No CSV files found")
    
    # Check for other important files
    other_files = [f for f in dataset_dir.glob("*") if f.is_file() and not (f.suffix == '.tif' or f.suffix == '.csv')]
    if other_files:
        print("\nOther files:")
        for file in other_files:
            print(f"  {file.name}")


def get_matching_files(dataset_dir: Path, sample_id: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find matching optical, SAR, and tiepoint files based on sample ID.
    
    Args:
        dataset_dir: Directory containing the dataset
        sample_id: Optional sample ID to find
        
    Returns:
        Tuple of (optical_path, sar_path, tiepoints_path)
    """
    # Check if dataset directory exists inside processed directory
    nested_dir = dataset_dir / dataset_dir.name
    if nested_dir.exists():
        dataset_dir = nested_dir
        
    # Find all relevant files
    tif_files = list(dataset_dir.glob("*.tif"))
    optical_files = [f for f in tif_files if "optical" in f.name.lower()]
    sar_files = [f for f in tif_files if "sar" in f.name.lower()]
    tiepoint_files = list(dataset_dir.glob("*tiepoints*.csv"))
    
    if not optical_files:
        print("No optical images found")
        return None, None, None
        
    if not sar_files:
        print("No SAR images found")
        return None, None, None
        
    # If no sample_id provided, use the first set
    if not sample_id:
        # Take first pair from the same area
        optical_path = optical_files[0]
        filename_parts = optical_path.name.split("_")
        prefix = filename_parts[0]  # e.g., "02" or "03"
        
        if "_train_" in optical_path.name:
            region = optical_path.name.split("_train_")[1].split(".")[0]
        else:
            region = "01"  # Default
        
        matching_sar = [f for f in sar_files if prefix in f.name and f"_train_{region}" in f.name]
        matching_tiepoints = [f for f in tiepoint_files if prefix in f.name and f"_train_{region}" in f.name]
        
        sar_path = matching_sar[0] if matching_sar else None
        tiepoints_path = matching_tiepoints[0] if matching_tiepoints else None
        
        return optical_path, sar_path, tiepoints_path
    
    # With sample_id provided, look for specific matches
    # First try direct filename matches
    matching_optical = [f for f in optical_files if sample_id in f.name]
    matching_sar = [f for f in sar_files if sample_id in f.name]
    matching_tiepoints = [f for f in tiepoint_files if sample_id in f.name]
    
    # If no direct matches, try to parse the sample ID for better matching
    if not matching_optical or not matching_sar:
        # Is it just a number like "01"?
        if sample_id.isdigit():
            # Look for any files with this number
            matching_optical = [f for f in optical_files if f"_train_{sample_id}" in f.name]
            matching_sar = [f for f in sar_files if f"_train_{sample_id}" in f.name]
            matching_tiepoints = [f for f in tiepoint_files if f"_train_{sample_id}" in f.name]
        
    if not matching_optical:
        print(f"No matching optical images found for '{sample_id}'")
        return None, None, None
        
    if not matching_sar:
        print(f"No matching SAR images found for '{sample_id}'")
        return None, None, None
    
    # Get first matches
    optical_path = matching_optical[0]
    sar_path = matching_sar[0]
    
    # Extract prefix (e.g., "02" or "03") from the optical file
    filename_parts = optical_path.name.split("_")
    prefix = filename_parts[0]
    
    # Extract region from optical file
    if "_train_" in optical_path.name:
        region = optical_path.name.split("_train_")[1].split(".")[0]
    else:
        region = sample_id  # Default
    
    # Find matching tiepoints file with same prefix and region
    matching_tiepoints = [f for f in tiepoint_files if prefix in f.name and region in f.name]
    tiepoints_path = matching_tiepoints[0] if matching_tiepoints else None
    
    return optical_path, sar_path, tiepoints_path


def visualize_sample(dataset_dir: Path, sample_id: Optional[str] = None) -> None:
    """Visualize a sample from the dataset (optical and SAR images with tiepoints if available).
    
    Args:
        dataset_dir: Directory containing the dataset
        sample_id: Sample ID to visualize (e.g., '01' or '02_train_01')
    """
    # Get matching files
    optical_path, sar_path, tiepoints_path = get_matching_files(dataset_dir, sample_id)
    
    if not optical_path or not sar_path:
        return
    
    # Load images
    print(f"Visualizing sample:")
    print(f"  Optical: {optical_path.name}")
    print(f"  SAR: {sar_path.name}")
    if tiepoints_path:
        print(f"  Tiepoints: {tiepoints_path.name}")
    
    with rasterio.open(optical_path) as src:
        optical_img = src.read()
        optical_img = np.moveaxis(optical_img, 0, -1)  # CHW -> HWC
        optical_img = np.clip(optical_img / np.percentile(optical_img, 98), 0, 1)
    
    with rasterio.open(sar_path) as src:
        sar_img = src.read()
        sar_img = np.squeeze(sar_img)  # Remove channel dimension if present
        sar_img = np.clip(sar_img / np.percentile(sar_img, 98), 0, 1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot optical image
    ax1.imshow(optical_img)
    ax1.set_title(f"Optical (RGB) - {optical_path.name}")
    ax1.axis('off')
    
    # Plot SAR image
    ax2.imshow(sar_img, cmap='gray')
    ax2.set_title(f"SAR - {sar_path.name}")
    ax2.axis('off')
    
    # Plot tiepoints if available
    if tiepoints_path:
        try:
            tiepoints_df = pd.read_csv(tiepoints_path)
            print(f"  Found {len(tiepoints_df)} tie-points")
            print(f"  Columns: {list(tiepoints_df.columns)}")
            
            # Check column names (the actual column names are sar_row, sar_col, optical_row, optical_col)
            if 'sar_col' in tiepoints_df.columns and 'optical_col' in tiepoints_df.columns:
                # Plot on optical image (x=col, y=row)
                ax1.scatter(
                    tiepoints_df['optical_col'], 
                    tiepoints_df['optical_row'], 
                    color='red', 
                    marker='+', 
                    s=50, 
                    alpha=0.7
                )
                
                # Plot on SAR image
                ax2.scatter(
                    tiepoints_df['sar_col'], 
                    tiepoints_df['sar_row'], 
                    color='red', 
                    marker='+', 
                    s=50, 
                    alpha=0.7
                )
                
                # Draw sample points with numbers
                for i in range(min(10, len(tiepoints_df))):
                    row = tiepoints_df.iloc[i]
                    ax1.annotate(str(i+1), (row['optical_col'], row['optical_row']), color='white', fontsize=10)
                    ax2.annotate(str(i+1), (row['sar_col'], row['sar_row']), color='white', fontsize=10)
            else:
                print(f"  Warning: Unknown tiepoint column format in {tiepoints_path.name}")
                
        except Exception as e:
            print(f"Error loading tiepoints: {e}")
    
    if sample_id:
        title = f"Sample: {sample_id}"
    else:
        title = f"Sample: {optical_path.stem}"
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to explore the SpaceNet 9 dataset."""
    parser = argparse.ArgumentParser(description="Explore SpaceNet 9 challenge data")
    parser.add_argument("--config", type=str, default="configs/data_urls.yaml", 
                        help="Path to configuration file with paths")
    parser.add_argument("--dataset", type=str, default="train", 
                        choices=["train", "publictest", "baseline-submission", 
                                "trivial-submission", "perfect-solution", "scorer"],
                        help="Dataset to explore")
    parser.add_argument("--sample_id", type=str, default=None, 
                        help="Specific sample ID to visualize (e.g., '01')")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--mode", choices=["list", "tree", "both"], default="both",
                       help="Exploration mode: list files, show directory tree, or both")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    processed_data_dir = Path(config["paths"]["processed"])
    dataset_dir = processed_data_dir / args.dataset
    
    # Explore directory structure based on mode
    if args.mode in ["tree", "both"]:
        explore_directory_structure(dataset_dir)
    
    if args.mode in ["list", "both"]:
        list_dataset_files(dataset_dir)
    
    # Visualize a sample unless told not to
    if not args.no_viz:
        visualize_sample(dataset_dir, args.sample_id)


if __name__ == "__main__":
    main() 