#!/bin/bash

# track simulation
echo "------------------------------------------------------"
echo "Generating events for learning optimal KL threshold..."
echo "------------------------------------------------------"

NUM_EVENTS=10000                # currently each event with 10 tracks
SIGMA0=0.5                  # r.m.s of track position measurements
EDGE_VAR_THRES=0.6         # remove nodes with mean edge orientation above threshold
OUTPUT=output/track_sim/sigma$SIGMA0/

mkdir -p $OUTPUT
echo "generating events..."
python generate_events.py -n $NUM_EVENTS -s $SIGMA0 -t $EDGE_VAR_THRES -o $OUTPUT

FILE_EXT=_events.gpickle
INPUT=$OUTPUT$NUM_EVENTS$FILE_EXT 
echo "extracting metadata..."
python extract_metadata.py -i $INPUT -o $OUTPUT -n $NUM_EVENTS

echo "DONE"