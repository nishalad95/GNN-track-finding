#!/bin/bash

# track simulation
echo "Running track simulation..."
mkdir -p output/track_sim/cca_output
python track_sim.py -t 0.6 -o output/track_sim/

# # iteration 1
mkdir -p output/iteration_1/outlier_removal
echo "Iteration 1 Running edge outlier removal & CCA..."
python clustering.py -i output/track_sim/cca_output/ -o output/iteration_1/outlier_removal/ -d track_state_estimates

mkdir output/iteration_1/track_candidates
mkdir output/iteration_1/remaining_network
mkdir output/iteration_1/cca_output
# # Calibration of KL cut
# # TODO: calibration of KL cut using MC truth
# # TODO: rerun outlier detection & clustering with new KL threshold
# echo "Running KF, extracting track candidates, executing CCA & updating track states..."
# python extract_track_candidates.py -i output/iteration_1/outlier_removal/ -c output/iteration_1/track_candidates/ -o output/iteration_1/cca_output/ -r output/iteration_1/remaining_network/ -cs 0.6

# # iteration 2
# mkdir -p output/iteration_2/updated
# mkdir output/iteration_2/outlier_removal
# mkdir output/iteration_2/track_candidates
# mkdir output/iteration_2/remaining_network
# mkdir output/iteration_2/cca_output
# echo "Iteration 2 Running edge outlier removal..."
# python extrapolate_merged_states.py -i output/iteration_1/cca_output/ -o output/iteration_2/updated/
# # Calibration of KL cut
# # TODO: calibration of KL cut using MC truth
# # TODO: rerun outlier detection & clustering with new KL threshold
# # python clustering.py -i output/iteration_2/updated/ -o output/iteration_2/ -d updated_track_state_estimates
# # detect and remove good track candidates using KF
# # rerun CCA
