#!/bin/bash

# track simulation
echo "Running track simulation..."
mkdir -p output/track_sim/subgraphs
python track_sim.py -t 0.6 -o output/track_sim/

# iteration 1
echo "Iteration 1 \nRunning edge outlier removal..."
mkdir -p output/iteration_1/subgraphs
python clustering.py -i output/track_sim/subgraphs/ -o output/iteration_1/
echo "Running Kalman Filter and extracting track candidates..."
mkdir -p output/iteration_1/track_candidates
mkdir -p output/iteration_1/remaining_network
python extract_track_candidates.py -i output/iteration_1/subgraphs/ -c output/iteration_1/track_candidates/ -r output/iteration_1/remaining_network/ -cs 0.6

# iteration 2
echo "Iteration 2 \nRunning edge outlier removal..."
mkdir -p output/iteration_2/subgraphs
python clustering.py -i output/iteration_1/remaining_network/ -o output/iteration_2/

