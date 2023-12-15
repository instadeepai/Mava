#!/bin/bash


# Loop 512 times
for i in {1..512}
do
   echo "Running iteration $i"
   python mava/systems/rec_ippo_rware.py
done
