#!/bin/bash

SIGMA0=0.5          # r.m.s measurement error
CS=0.2              # p-value acceptance level for good track candidate extraction
n=4                 # minimum number of hits for good track candidate acceptance
LUT=learn_KL/output/empvar/empvar.lut
ROOTDIR=output

# track simulation
echo "----------------------------"
echo "Running track simulation..."
echo "----------------------------"
OUTPUT=$ROOTDIR/track_sim/
mkdir -p ${OUTPUT}cca_output
python track_sim.py -t 0.7 -o $OUTPUT -e $SIGMA0


# iteration 1
echo "----------------"
echo "Iteration 1"
echo "------------------------------------------------------------------"
echo "Running clusterization/outlier removal --> merging track states..."
echo "------------------------------------------------------------------"
INPUT=$ROOTDIR/track_sim/cca_output/
OUTPUT=$ROOTDIR/iteration_1/outlier_removal/
mkdir -p $OUTPUT
# clustering with empirical variance
python clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT


# EXTRACT GOOD CANDIDATES
echo "------------------------------------------------------------------"
echo "Extracting potential good track candidates, executing CCA & KF..."
echo "------------------------------------------------------------------"
INPUT=$ROOTDIR/iteration_1/outlier_removal/
CANDIDATES=$ROOTDIR/iteration_1/candidates/
REMAINING=$ROOTDIR/iteration_1/remaining/
mkdir -p $CANDIDATES
mkdir -p $REMAINING
python extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -cs $CS -e $SIGMA0 -n $n


# iteration 2
echo "--------------"
echo "Iteration 2"
echo "-------------------------------------------------------------------"
echo "Running message passing, extrapolation & validation..."
echo "-------------------------------------------------------------------"
INPUT=$REMAINING
OUTPUT=$ROOTDIR/iteration_2/extrapolated/
mkdir -p $OUTPUT
python extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c 10


# EXTRACT GOOD CANDIDATES
echo "------------------------------------------------------------------"
echo "Extracting potential good track candidates, executing CCA & KF..."
echo "------------------------------------------------------------------"
INPUT=$OUTPUT
CANDIDATES=$ROOTDIR/iteration_2/candidates/
REMAINING=$ROOTDIR/iteration_2/remaining/
mkdir -p $CANDIDATES
mkdir -p $REMAINING
cp -r output/iteration_1/candidates/ $CANDIDATES
python extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -cs $CS -e $SIGMA0 -n $n


# iteration 3
echo "----------------"
echo "Iteration 3"
echo "-------------------------------------------------------------"
echo "Running clustering/outlier masking on updated track states"
echo "-------------------------------------------------------------"
INPUT=$REMAINING
OUTPUT=$ROOTDIR/iteration_3/outlier_removal/
mkdir -p $OUTPUT
# clustering with empirical variance
python clustering_updated_states.py -i $INPUT -o $OUTPUT -d updated_track_states -l $LUT


# EXTRACT GOOD CANDIDATES
echo "------------------------------------------------------------------"
echo "Extracting potential good track candidates, executing CCA & KF..."
echo "------------------------------------------------------------------"
INPUT=$OUTPUT
CANDIDATES=$ROOTDIR/iteration_3/candidates/
REMAINING=$ROOTDIR/iteration_3/remaining/
mkdir -p $CANDIDATES
mkdir -p $REMAINING
cp -r output/iteration_2/candidates/ $CANDIDATES
python extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -cs $CS -e $SIGMA0 -n $n


# iteration 4
echo "--------------"
echo "Iteration 4"
echo "-------------------------------------------------------------------"
echo "Running message passing, extrapolation & validation..."
echo "-------------------------------------------------------------------"
INPUT=$REMAINING
OUTPUT=$ROOTDIR/iteration_4/extrapolated/
mkdir -p $OUTPUT
python extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c 5


# EXTRACT GOOD CANDIDATES
echo "------------------------------------------------------------------"
echo "Extracting potential good track candidates, executing CCA & KF..."
echo "------------------------------------------------------------------"
INPUT=$OUTPUT
CANDIDATES=$ROOTDIR/iteration_4/candidates/
REMAINING=$ROOTDIR/iteration_4/remaining/
mkdir -p $CANDIDATES
mkdir -p $REMAINING
cp -r output/iteration_3/candidates/ $CANDIDATES
python extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -cs $CS -e $SIGMA0 -n $n



echo "DONE"