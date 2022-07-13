#!/bin/bash

i=0
while [ "$i" -le "$2" ]; do
    nohup python3 ./src/Optuna.py -f $1 > /dev/null 2>&1 &
    i=$(($i + 1))
done

echo "Done!"