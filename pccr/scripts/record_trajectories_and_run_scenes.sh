

bash scripts/record_trajectories_wrapper.sh

python3 core/prune_trajectories.py \
    --input output/trajectories \
    --output output/trajectories_pruned \
    --scenes configs/scenes.json \
    --total trainval:80 \
    --total test:30 \
    --total mini:5 \
    --smoothness 1.0 

bash scripts/run_scenes_wrapper.sh