#!/bin/bash


EDGE_VAR_THRES=0.6         # remove nodes with mean edge orientation above threshold

echo "extracting metadata..."
python extract_metadata.py -i $INPUT -o $OUTPUT -n $NUM_EVENTS

echo "DONE"