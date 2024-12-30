#!/bin/bash

COMMAND="./test"

for i in {1..100}
do
    echo "Running Iteration $i"
    $COMMAND >> testHolderAllThree.txt
done
