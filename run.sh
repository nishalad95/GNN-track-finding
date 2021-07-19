#!/bin/bash

# track simulation
echo "----------------------------"
echo "Running track simulation..."
echo "----------------------------"
mkdir -p output/track_sim/cca_output
python track_sim.py -t 0.6 -o output/track_sim/


# iteration 1
echo "----------------"
echo "Iteration 1"
echo "-------------------------------------------------------------"
echo "Running edge outlier removal, CCA & calibration of KL cut..."
echo "-------------------------------------------------------------"
mkdir -p output/iteration_1/outlier_removal
# clustering with degree of node
# python clustering.py -i output/track_sim/cca_output/ -o output/iteration_1/outlier_removal/ -d track_state_estimates -l learn_KL/output/kl_dist_degree/kl_degree.lut
# clustering with empirical variance
python clustering.py -i output/track_sim/cca_output/ -o output/iteration_1/outlier_removal/ -d track_state_estimates -l learn_KL/output/kl_dist_vs_emp_var/kl_empvar.lut


# iteration 2
echo "--------------"
echo "Iteration 2"
echo "---------------------------------------------------------------"
echo "Running track extrapolation & reweighting of track states..."
echo "---------------------------------------------------------------"
mkdir -p output/iteration_2/extrapolated
python extrapolate_merged_states.py -i output/iteration_1/outlier_removal/ -o output/iteration_2/extrapolated/


# mkdir output/iteration_1/track_candidates
# mkdir output/iteration_1/remaining_network
# mkdir output/iteration_1/cca_output
# echo "-------------------------------------------------------------"
# echo "Running KF, extracting track candidates, executing CCA..."
# echo "-------------------------------------------------------------"
# python extract_track_candidates.py -i output/iteration_1/outlier_removal/ -c output/iteration_1/track_candidates/ -o output/iteration_1/cca_output/ -r output/iteration_1/remaining_network/ -cs 0.6


# # UP TO HERE

# echo "---------------------------------------------------------------"
# echo "Running edge outlier removal, CCA & calibration of KL cut..."
# echo "---------------------------------------------------------------"
# mkdir output/iteration_2/outlier_removal
# python clustering.py -i output/iteration_2/extrapolated/ -o output/iteration_2/outlier_removal/ -d updated_track_state_estimates



# echo "-------------------------------------------------------------"
# echo "Running KF, extracting track candidates, executing CCA..."
# echo "-------------------------------------------------------------"
# mkdir output/iteration_2/track_candidates
# mkdir output/iteration_2/remaining_network
# mkdir output/iteration_2/cca_output

echo "DONE"