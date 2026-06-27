#!/bin/bash

# Configuration
PYTHON="/usr/bin/python"
DATA_ROOT="/workspace/output/data"
TARGET_ROOT="" # leave empty if you want to save in the same folder and be flexible with data loading
VERSIONS=("v1.0-mini" "v1.0-trainval" "v1.0-test")

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Define the folders to process
FOLDERS=("R1" "R1-c6" "R1-c10" "R1-f" "R1-r" "R1-t" "R2" "R3" "R4" "R5" "R6" "R7" "R8" "R9")

for folder in "${FOLDERS[@]}"; do
    dir="$DATA_ROOT/$folder"
    for version in "${VERSIONS[@]}"; do
        if [[ -d "$dir/$version" ]]; then
            echo -e "${GREEN}Processing dataset in: $dir${NC}"
            
            $PYTHON ./tools/data/create_data.py \
                --save-folder "$dir" \
                --root-path "$TARGET_ROOT" \
                --version "$version" \
                --no-lidar
        else
            echo "Skipping $version in $dir (no nuScenes version folders found)"
        fi
    done
done

echo -e "${GREEN}All datasets processed successfully.${NC}"
