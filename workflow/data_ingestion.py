import click
import zipfile
from pathlib import Path
import time
import os
import shutil


@click.command()
@click.option('--input_zip', default='/data_gem/raw_data/2021_data.zip')
def main(input_zip):
    """ 
    This script is used to automatically update the base dataset (raw_data) with a zipped folder containing new data into the right locations.
    The folder structure should be:
    FOLDER NAME 
    ├── emerald
    │   └── client_stones
    │       ├── ed
    │       ├── ftir
    │       ├── master_files
    │       └── uv
    ├── ruby
    │   └── client_stones
    │       ├── ed
    │       ├── ftir
    │       ├── master_files
    │       └── uv
    └── sapphire
        └── client_stones
            ├── ed
            ├── ftir
            ├── master_files
            └── uv
            
    Before starting make sure that you have made a backup of the raw_data folder + master_file.h5
    """
    print("WARNING: The script must be run as sudo")
    resp = input("Have you backed up the raw_data folder + master_file.h5? (y/n)")
    if resp.lower() != "y":
        print("Exiting...")
        exit()
    # Unzip the zip file
    start_time = time.time()
    dir_for_unzip = Path(input_zip).parent
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall(dir_for_unzip)
        
    # Get the list of files in the zip file by looking at all files created after the start time
    files = []
    for x in dir_for_unzip.glob("*"):
        created_time = os.path.getctime(x)
        if created_time > start_time:
            files.append(x)
    assert len(files) == 1, "There should be only one folder after unzipping"
    
    stone_types = ["sapphire","ruby","emerald"]
    data_sources = ["ed","ftir","uv","master_files","icp_single_files"]
    for stone_type in stone_types:
        print(f"Processing {stone_type}")
        base_folder = files[0] / stone_type / "client_stones"
        if not base_folder.exists():
            print(f"{base_folder} does not exist")
            continue
        for data_source in data_sources:
            print(f"\t Processing {data_source}")
            src_folder = base_folder / data_source
            # Copy all files from the src_folder to the raw_data folder 
            target_dir = Path("/data_gem/raw_data") / stone_type / "client_stones" / data_source
            
            for file in src_folder.glob("*"):
                name_file = file.name
                shutil.copy(str(file), str(target_dir / name_file)) 
    
    
    
    
if __name__ == "__main__":
    main()