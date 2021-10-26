#!/bin/bash

NUM_SIMS=30                
OUTPUT_FILE=post_iteration2_reweights.csv


for i in $(seq 1 $NUM_SIMS);
    do
        echo "running sim ${i}..."
        sh ./run_gnn.sh > run_gnn.std.out
        
        echo "extracting truth & edge reweightings.."
        python extract_remaining_edge_reweights.py
        
        echo "deleting output directory"
        rm -rf output/*
done

echo "DONE"