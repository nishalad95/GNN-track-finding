#!/bin/bash

SIGMA0=0.5
LUT=learn_KL/output/empvar/empvar.lut

# track simulation
echo "----------------------------"
echo "Running track simulation..."
echo "----------------------------"
OUTPUT=output/track_sim/
mkdir -p ${OUTPUT}cca_output
python track_sim.py -t 0.6 -o $OUTPUT -e $SIGMA0

# # OLD CLUSTERING ALG
# # iteration 1
# echo "----------------"
# echo "Iteration 1"
# echo "------------------------------------------------------------------"
# echo "Running clusterization/outlier removal --> merging track states..."
# echo "------------------------------------------------------------------"
# INPUT=output/track_sim/cca_output/
# OUTPUT=output_old/iteration_1/outlier_removal/
# mkdir -p $OUTPUT
# # clustering with empirical variance
# python clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l learn_KL/output/empvar/empvar.lut

# NEW CLUSTERING ALG
# iteration 1
echo "----------------"
echo "Iteration 1"
echo "------------------------------------------------------------------"
echo "Running clusterization/outlier removal --> merging track states..."
echo "------------------------------------------------------------------"
INPUT=output/track_sim/cca_output/
OUTPUT=output/iteration_1/outlier_removal/
mkdir -p $OUTPUT
# clustering with empirical variance
python clustering_v2.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT

# iteration 2
echo "--------------"
echo "Iteration 2"
echo "-------------------------------------------------------------------"
echo "Running message passing, extrapolation & validation..."
echo "-------------------------------------------------------------------"
INPUT=output/iteration_1/outlier_removal/
OUTPUT=output/iteration_2/extrapolated/
mkdir -p $OUTPUT
python extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c 10

# iteration 3
echo "----------------"
echo "Iteration 3"
echo "-------------------------------------------------------------"
echo "Running clustering/outlier masking on updated track states"
echo "-------------------------------------------------------------"
INPUT=output/iteration_2/extrapolated/
OUTPUT=output/iteration_3/outlier_removal
mkdir -p $OUTPUT
# clustering with empirical variance
python clustering_v2_updated_states.py -i $INPUT -o $OUTPUT -d updated_track_states -l $LUT


# # iteration 4
# echo "--------------"
# echo "Iteration 4"
# echo "-------------------------------------------------------------------"
# echo "Running message passing, extrapolation & validation..."
# echo "-------------------------------------------------------------------"
# INPUT=output/iteration_3/outlier_removal
# OUTPUT=output/iteration_4/extrapolated/
# mkdir -p $OUTPUT
# python extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c 5


# mkdir output/iteration_1/track_candidates
# mkdir output/iteration_1/remaining_network
# mkdir output/iteration_1/cca_output
# echo "-------------------------------------------------------------"
# echo "Running KF, extracting track candidates, executing CCA..."
# echo "-------------------------------------------------------------"
# python extract_track_candidates.py -i output/iteration_1/outlier_removal/ -c output/iteration_1/track_candidates/ -o output/iteration_1/cca_output/ -r output/iteration_1/remaining_network/ -cs 0.6 -e $SIGMA0


# echo "-------------------------------------------------------------"
# echo "Running KF, extracting track candidates, executing CCA..."
# echo "-------------------------------------------------------------"
# mkdir output/iteration_2/track_candidates
# mkdir output/iteration_2/remaining_network
# mkdir output/iteration_2/cca_output

echo "DONE"