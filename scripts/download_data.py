#!/usr/bin/env python
"""
Download SpaceNet 9 challenge data.
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm


def download_file(url: str, destination: Path, force: bool = False) -> bool:
    """Download a file from a URL to a local destination with progress bar.
    
    Args:
        url: The URL to download from
        destination: The local path to save the file
        force: If True, download even if file already exists
        
    Returns:
        True if download was successful, False otherwise
    """
    # Check if file already exists
    if destination.exists() and not force:
        print(f"File already exists at {destination}. Skipping download. Use --force to re-download.")
        return True
        
    print(f"Downloading {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        import requests
        from tqdm.auto import tqdm
        
        # Streaming download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Get file size if available
        total = int(response.headers.get('content-length', 0))
        
        # Initialize progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    bar.update(size)
                    
        print(f"Successfully downloaded to {destination}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main() -> None:
    """Main function to download SpaceNet 9 data."""
    parser = argparse.ArgumentParser(description="Download SpaceNet 9 challenge data")
    parser.add_argument("--config", type=str, default="configs/data_urls.yaml", 
                        help="Path to configuration file with URLs")
    parser.add_argument("--datasets", nargs="+", 
                        choices=["train", "public_test", "baseline_submission", 
                                "trivial_submission", "perfect_solution", "scorer", "all"],
                        default=["all"], help="Datasets to download")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-download even if files already exist")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    urls: Dict[str, str] = config["urls"]
    raw_data_dir: Path = Path(config["paths"]["raw"])
    
    # Determine which datasets to download
    datasets_to_download: List[str] = list(urls.keys()) if "all" in args.datasets else args.datasets
    
    # Download each dataset
    for dataset in datasets_to_download:
        url = urls.get(dataset)
        if not url:
            print(f"Warning: URL for {dataset} not found in config file")
            continue
            
        filename = url.split("/")[-1]
        destination = raw_data_dir / filename
        
        download_file(url, destination, force=args.force)
    
    print("Download process completed")


if __name__ == "__main__":
    main() 