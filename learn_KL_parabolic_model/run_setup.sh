#!/bin/bash

# This script will generate N number of events and extract the relevant metadata to be
# used for training a classifier to learn the optimal KL threshold
# This script needs to be run from the directory: learn_KL/parabolic_model/


# track simulation
echo "---------------------------------------------------------------------------------"
echo "Generating events for learning optimal KL threshold with the parabolic model..."
echo "---------------------------------------------------------------------------------"

NUM_EVENTS=1000               # currently each event with 10 tracks
SIGMA0=4.0                  # ignore this, uncertainties are properly defined in compute_track_state_estimates
EDGE_VAR_THRES=0.8         # remove nodes with mean edge orientation above threshold
OUTPUT=output/track_sim/sigma$SIGMA0/

# generate and save N number of events and save data in a .gpickle file
mkdir -p $OUTPUT
echo "generating events..."
python generate_training_data/generate_events.py -n $NUM_EVENTS -s $SIGMA0 -t $EDGE_VAR_THRES -o $OUTPUT

# # extract the relevant metadata for training the classifier: 'kl_dist', 'emp_var', 'degree', 'truth'
# # save this data in a csv file
INPUT=$OUTPUT 
echo "extracting metadata..."
python generate_training_data/extract_metadata.py -i $INPUT -o $OUTPUT -n $NUM_EVENTS

# plot the training data
# python SVM_training_predictions/plot_training_data.py

echo "DONE"