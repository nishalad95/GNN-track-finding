#!/bin/bash


for i in {1..50}
do
   echo "Iteration $i, running gnn algorithm"
   sh ./run_gnn.sh
   echo "running agglomerative clustering & saving maximum distance values"
   python shared_hit_identification/weight_v_angle_dist_stats.py
   echo "Deleting directory"
   rm -r output/

done