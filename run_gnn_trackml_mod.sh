#!/bin/bash

# ---------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------
# iterations
START=1
END=2

# event conversion and track simulation
min_volume=7            # minimum volume number to analyse (inclusive) - used also in effciency calc
max_volume=9            # maximum volume number to analyse (inclusive) - used also in efficiency calc
SIGMA0=0.1              # IMPORTANT: sigma0 is used in the extraction KF only, r.m.s measurement error in xy plane
SIGMA_MS=0.01         # 10^-4 multiple scattering error
ROOTDIR=src/output      # output directory to store GNN algorithm output

# clustering
# LUT=learn_KL_linear_model/output/empvar/empvar_relaxed.lut   # LUT file for KL distance calibration
LUT=learn_KL_linear_model/output/empvar/empvar.lut

# extrapolation
c=1                     #  initial chi2 distance acceptance threshold for extrapolated states

# extracting track candidates
p=0.01                  # p-value acceptance level for good track candidate extraction - currently applied in xy plane
n=4                     # minimum number of hits for good track candidate acceptance (>=n)
s=10                    # 3d distance threshold for close proximity nodes, used in KF rotatation if nodes too close together
# used in node-merging in extraction for close proximity nodes, but this will change due to PDA
t=8.0                   # threshold distance node merging in extraction
# ----------------------------------------------------------------------------------------------


# -----------------------------------------------------
# Profiling setup
# -----------------------------------------------------
stages=("start_time")
time=$SECONDS
execution_times=($time)

# Comment out for debugging when event-to-network conversion is not needed!!
# -----------------------------------------------------
# Event conversion into graph network
# -----------------------------------------------------
mkdir -p $ROOTDIR
echo "----------------------------------------------------"
echo "Running conversion of generated events to GNN..."
echo "----------------------------------------------------"
INPUT=$ROOTDIR/track_sim/network/
mkdir -p $INPUT
EVENT_NETWORK=src/trackml_mod/event_network/minCurv_0.3_800
EVENT_TRUTH=src/trackml_mod/event_truth
python src/trackml_mod/event_conversion.py -o $INPUT -e $SIGMA0 -m $SIGMA_MS -n $EVENT_NETWORK -t $EVENT_TRUTH -a $min_volume -z $max_volume
stages+=("event_conversion")
time=$SECONDS
execution_times+=($time)

echo "------------------------------------------------------"
echo "Extracting track candidates post CCA: iteration0"
echo "------------------------------------------------------"
INPUT=$ROOTDIR/track_sim/network/
CANDIDATES=$ROOTDIR/iteration_0/candidates/
REMAINING=$ROOTDIR/iteration_0/remaining/
FRAGMENTS=$ROOTDIR/iteration_0/fragments/
mkdir -p $CANDIDATES
mkdir -p $REMAINING
mkdir -p $FRAGMENTS
python src/extract/extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -f $FRAGMENTS -p $p -e $SIGMA0 -m $SIGMA_MS -n $n -s $s -t $t -a 0


# -----------------------------------------------------
# Begin the iterations....
# -----------------------------------------------------
INPUT=$REMAINING
for (( i=$START; i<=$END; i++ ))

# INPUT=$ROOTDIR/iteration_1/remaining/
# for (( i=2; i<=2; i++ ))

    do
        OUTPUT=$ROOTDIR/iteration_$i/network/
        mkdir -p $OUTPUT

        if (( $i == 1 ))
        then
            echo "----------------------------------------------------"
            echo "Iteration ${i}: GMR via clusterisation"
            echo "----------------------------------------------------"
            prev_duration=$SECONDS
            python src/clustering/clustering.py -i $INPUT -o $OUTPUT -d track_state_estimates -l $LUT
            # time it!
            stages+=("clustering")
            time=$SECONDS
            execution_times+=($time)
        elif (( $i % 2 == 0 ))
        then
            echo "---------------------------------------------------"
            echo "Iteration ${i}: Neighbourhood Aggregation"
            echo "---------------------------------------------------"
            echo "Using chisq distance cut of: ${c}"
            prev_duration=$SECONDS
            python src/extrapolate/extrapolate_merged_states.py -i $INPUT -o $OUTPUT -c $c -m $SIGMA_MS
            # let c=$c*0.5   # tighter cut each time

            # time it!
            stages+=("extrapolation")
            time=$SECONDS
            execution_times+=($time)
        fi

        echo "------------------------------------------------------"
        echo "Extracting track candidates"
        echo "------------------------------------------------------"
        INPUT=$OUTPUT
        CANDIDATES=$ROOTDIR/iteration_$i/candidates/
        REMAINING=$ROOTDIR/iteration_$i/remaining/
        FRAGMENTS=$ROOTDIR/iteration_$i/fragments/
        mkdir -p $CANDIDATES
        mkdir -p $REMAINING
        mkdir -p $FRAGMENTS
        # if (( $i > 1 ))
        # then
        let num=$i-1
        cp -r $ROOTDIR/iteration_$num/candidates/ $CANDIDATES
        # fi
        python src/extract/extract_track_candidates.py -i $INPUT -c $CANDIDATES -r $REMAINING -f $FRAGMENTS -p $p -e $SIGMA0 -m $SIGMA_MS -n $n -s $s -t $t -a $i
        INPUT=$REMAINING

        # time it!
        stages+=("extract")
        time=$SECONDS
        execution_times+=($time)

done


echo "----------------------------------------------------"
# echo "Running track reconstruction efficiency:"
python src/extract/reconstruction_efficiency.py -t $EVENT_TRUTH -o $ROOTDIR -a $min_volume -z $max_volume -i $END
echo "Plotting Purity distribution..."
python src/extract/purity_distribution.py -i $ROOTDIR
echo "Plotting p-value distribution..."
python src/extract/p_value_distribution.py -i $ROOTDIR
echo "----------------------------------------------------"


# time it!
stages+=("reconstruction_efficiency")
time=$SECONDS
execution_times+=($time)

# plot all candidates
python src/extract/plot_all_extracted_candidates.py -i $END

# time it!
stages+=("plot_all_candidates")
time=$SECONDS
execution_times+=($time)


echo "----------------------------------------------------"
echo "Execution stages and times:"
printf "%s\n" "${stages[@]}" > $ROOTDIR/execution_stages.txt
for value in "${stages[@]}"
do
     echo "$value"
done

printf "%s\n" "${execution_times[@]}" > $ROOTDIR/execution_times.txt
for value in "${execution_times[@]}"
do
     echo "$value"
done
echo "----------------------------------------------------"


echo "DONE"