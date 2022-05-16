#!/bin/bash

count=0
total=0
while read l; do
  touch "$2/$l-console.log"
  nohup python3 ./src/main.py -f $1 -lrb "$2/$l" --frequency 5 --seed "$l" >> "$2/$l-console.log" &
  count=$((count+1))
  if [ $count -eq 10 ]; then
      count=0
      wait
      total=$((total + 10))
      echo "$total Complete..."
  fi
done <./resources/seeds_long.list
echo "Done!"