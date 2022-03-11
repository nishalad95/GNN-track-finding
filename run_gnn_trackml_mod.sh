#!/bin/bash

# ---------------------------------------------------------------------------------------------
# VARIABLES
# track sim
VAR=100  #(don't remove any nodes)   # TEMPORARY: remove nodes with empirical variance greater than VAR
SIGMA0=0.0001                           # r.m.s measurement error 100 microns in xy plane
SIGMA_MS=0.0001                       # 10^-4 multiple scattering error
ROOTDIR=src/output

# clustering
LUT=learn_KL/output/empvar/empvar.lut       # LUT file for KL distance calibration

# extracting track candidates
p=0.01                  # p-value acceptance level for good track candidate extraction
n=4                     # minimum number of hits for good track candidate acceptance (>=n)

# extrapolation
c=2  # initial chisquare distance acceptance threshold factor for extrapolated states
# ----------------------------------------------------------------------------------------------


# mkdir -p $ROOTDIR
# # track conversion
# echo "-------------------------------------------------"
# echo "Running conversion of generated events to GNN..."
# echo "-------------------------------------------------"
# start_conversion=$SECONDS
# INPUT=$ROOTDIR/track_sim/network/
# mkdir -p $INPUT

# # event information
# EVENT_NETWORK=src/trackml_mod/event_network/minCurv_0.3_134
EVENT_TRUTH=src/trackml_mod/event_truth

# python src/trackml_mod/event_conversion.py -o $INPUT -e $SIGMA0 -m $SIGMA_MS -n $EVENT_NETWORK -t $EVENT_TRUTH
# conversion_duration=$(( SECONDS - start_conversion ))
# echo "Execution time event_conversion.py: $conversion_duration seconds"



# # copy the first 100 files over - DEVELOPMENT ONLY
# mkdir $ROOTDIR/track_sim/network_100/
# ls $ROOTDIR/track_sim/network/* | head -500 | xargs -I{} cp {} $ROOTDIR/track_sim/network_100/



INPUT=$ROOTDIR/track_sim/network_100/
# time it!
start=$SECONDS
execution_times=($start)
stages=("start")



# TESTING FOR ITERATION 2 ONLY
# INPUT=$ROOTDIR/iteration_1/remaining/       # THIS LINE CAN BE REMOVED
# for i in {2..2};                            # THIS LINE CAN BE REMOVED

for i in {1..1};
    do
        OUTPUT=$ROOTDIR/iteration_$i/network/
        mkdir -p $OUTPUT

        if (( $i == 1 ))
        then
            echo "-------------------------------------------------"
            echo "Iteration ${i}: Clusterization/Outlier Removal"
            echo "-------------------------------------------------"
            prev_duration=$SECONDS
            python src/clustering/clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT -e $SIGMA0
            # time it!
            prev_duration=$(( SECONDS - prev_duration ))
            echo "-------------------------------------------------"
            echo "Execution time, clustering.py: $prev_duration seconds"
            echo "-------------------------------------------------"
            execution_times+=($prev_duration)
            stages+=("clustering.py")

        elif (( $i % 2 == 0 ))
        then
            echo "------------------------------------------------"
            echo "Iteration ${i}: Message passing & Extrapolation"
            echo "------------------------------------------------"
            echo "Using chisq distance cut of: ${c}"
            prev_duration=$SECONDS
            python src/extrapolate/extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c $c -m $SIGMA_MS
            let c=$c/2   # tighter cut each time

            # time it!
            prev_duration=$(( SECONDS - prev_duration ))
            echo "-------------------------------------------------"
            echo "Execution time, extrapolate_merged_states.py: $prev_duration seconds"
            echo "-------------------------------------------------"
            execution_times+=($prev_duration)
            stages+=("extrapolate_merged_states.py")

        elif (( $i == 3 ))
        then
            echo "----------------------------------------------------------------------------"
            echo "Iteration ${i}: Clusterization on remaining network, reactivating all edges"
            echo "----------------------------------------------------------------------------"
            prev_duration=$SECONDS
            python src/clustering/clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT -e $SIGMA0 -r True

            # time it!
            prev_duration=$(( SECONDS - prev_duration ))
            echo "-----------------------------------------------------------------------------"
            echo "Execution time, clustering.py reactivating all edges: $prev_duration seconds"
            echo "-----------------------------------------------------------------------------"
            stages+=("clustering.py reactivating edges")

        else #(( $i % 2 == 1 ))
            echo "------------------------------------------------"
            echo "Iteration ${i}: Clusterization on updated states"
            echo "------------------------------------------------"
            prev_duration=$SECONDS
            python src/clustering/clustering_updated_states.py -i $INPUT -o $OUTPUT -d updated_track_states -l $LUT

            # time it!
            prev_duration=$(( SECONDS - prev_duration ))
            echo "--------------------------------------------------------------------"
            echo "Execution time, clustering_updated_states.py: $prev_duration seconds"
            echo "--------------------------------------------------------------------"
            execution_times+=($prev_duration)
            stages+=("clustering_updated_states.py")
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
        prev_duration=$SECONDS
        python src/extract/extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -p $p -e $SIGMA0 -m $SIGMA_MS -n $n
        INPUT=$REMAINING

        # time it!
        prev_duration=$(( SECONDS - prev_duration ))
        echo "---------------------------------------------------------------------"
        echo "Execution time, extract_track_candidates.py: $prev_duration seconds"
        echo "---------------------------------------------------------------------"
        execution_times+=($prev_duration)
        stages+=("extract_track_candidates.py")


done

end_duration=$(( SECONDS - start ))
echo "-------------------------------------------------"
echo "Execution time, entire GNN algorithm: $end_duration seconds"
echo "-------------------------------------------------"
execution_times+=($end_duration)
stages+=("end_duration")

# save execution times to file
printf "%s\n" "${execution_times[@]}" > execution_times/execution_times.txt
for value in "${execution_times[@]}"
do
     echo $value
done

printf "%s\n" "${stages[@]}" > execution_times/stages.txt
for value in "${stages[@]}"
do
     echo $value
done



echo "-------------------------------------------------"
echo "Running track reconstruction efficiency:"
echo "-------------------------------------------------"
python src/extract/reconstruction_efficiency.py -t $EVENT_TRUTH -o $ROOTDIR


echo "DONE"