#!/bin/bash

# ---------------------------------------------------------------------------------------------
# VARIABLES
# track sim
VAR=100  #(don't remove any nodes)   # TEMPORARY: remove nodes with empirical variance greater than VAR
SIGMA0=0.5                    # r.m.s measurement error
ROOTDIR=trackml_mod/output                # main output directory to save results of algorithm

trackml_mod=1

# clustering
LUT=learn_KL/output/empvar/empvar.lut       # LUT file for KL distance calibration

# extracting track candidates
p=0.01                  # p-value acceptance level for good track candidate extraction
n=3                     # minimum number of hits for good track candidate acceptance

# extrapolation
c=2  # initial chisquare distance acceptance threshold factor for extrapolated states
# ----------------------------------------------------------------------------------------------


# # track simulation
# echo "----------------------------"
# echo "Running track simulation..."
# echo "----------------------------"
OUTPUT=$ROOTDIR/track_sim/
INPUT=$ROOTDIR/track_sim/network_10/

for i in {1..2};
    do
        OUTPUT=$ROOTDIR/iteration_$i/network/
        mkdir -p $OUTPUT

        if (( $i == 1 ))
        then
            echo "-------------------------------------------------"
            echo "Iteration ${i}: Clusterization/Outlier Removal"
            echo "-------------------------------------------------"
            python clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT -e $SIGMA0
        elif (( $i % 2 == 0 ))
        then
            echo "------------------------------------------------"
            echo "Iteration ${i}: Message passing & Extrapolation"
            echo "------------------------------------------------"
            echo "Using chisq distance cut of: ${c}"
            python extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c $c
            let c=$c/2   # tighter cut each time
        elif (( $i == 3 ))
        then
            echo "----------------------------------------------------------------------------"
            echo "Iteration ${i}: Clusterization on remaining network, reactivating all edges"
            echo "----------------------------------------------------------------------------"
            python clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT -e $SIGMA0 -r True
        else #(( $i % 2 == 1 ))
            echo "------------------------------------------------"
            echo "Iteration ${i}: Clusterization on updated states"
            echo "------------------------------------------------"
            python clustering_updated_states.py -i $INPUT -o $OUTPUT -d updated_track_states -l $LUT
        fi
        
        # TODO: Run shared_hit_identification here!



        echo "---------------------------------"
        echo "Extracting good track candidates"
        echo "---------------------------------"
        INPUT=$OUTPUT
        CANDIDATES=$ROOTDIR/iteration_$i/candidates/
        REMAINING=$ROOTDIR/iteration_$i/remaining/
        mkdir -p $CANDIDATES
        mkdir -p $REMAINING
        if (( $i > 1 ))
        then
            let num=$i-1
            cp -r $ROOTDIR/iteration_$num/candidates/ $CANDIDATES
        fi
        python extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -p $p -e $SIGMA0 -t $trackml_mod -n $n
        INPUT=$REMAINING
done
echo "DONE"