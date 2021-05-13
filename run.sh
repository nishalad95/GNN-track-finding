#!/bin/bash

# track simulation
echo "Running track simulation..."
mkdir -p output/track_sim/subgraphs
python track_sim.py -t 0.5 -o output/track_sim/

# iteration 1
echo "Iteration 1 Running edge outlier removal..."
mkdir -p output/iteration_1/subgraphs
python clustering.py -i output/track_sim/subgraphs/ -o output/iteration_1/ -d track_state_estimates
echo "Running Kalman Filter and extracting track candidates..."
mkdir -p output/iteration_1/track_candidates
mkdir -p output/iteration_1/remaining_network
python extract_track_candidates.py -i output/iteration_1/subgraphs/ -c output/iteration_1/track_candidates/ -r output/iteration_1/remaining_network/ -cs 0.6

# iteration 2
echo "Iteration 2 Running edge outlier removal..."
mkdir -p output/iteration_2/updated
mkdir -p output/iteration_2/subgraphs
python extrapolate_merged_states.py -i output/iteration_1/remaining_network/ -o output/iteration_2/updated/
python clustering.py -i output/iteration_2/updated/ -o output/iteration_2/ -d updated_track_state_estimates


# echo "Running Kalman Filter and extracting track candidates..."
# mkdir -p output/iteration_2/track_candidates
# mkdir -p output/iteration_2/remaining_network
# python extract_track_candidates.py -i output/iteration_2/subgraphs/ -c output/iteration_2/track_candidates/ -r output/iteration_2remaining_network/ -cs 0.6
