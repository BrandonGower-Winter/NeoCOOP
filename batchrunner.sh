#!/bin/bash

count=0
limit=10
time=60
while read l; do
  touch "$2/$l-console.log"
  nohup python3 ./src/main.py -f $1 -lrb "$2/$l" --frequency 5 --seed "$l" >> "$2/$l-console.log" &
  sleep .5
  count=$(ps | grep -c "python3")
  while [ $count -ge $limit ]; do
      sleep $time
      count=$(ps | grep -c "python3")
  done
done <./resources/seeds_long.list
echo "Done!"