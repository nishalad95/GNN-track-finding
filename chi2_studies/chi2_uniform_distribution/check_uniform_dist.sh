#!/bin/bash

for i in {1..100}
do
   echo "Iteration $i, saving p-values"
   sh ./run_gnn.sh
   echo "Deleting directory"
   rm -r uniform-cs/

done