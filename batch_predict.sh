#!/bin/zsh

i=0
for eachfile in ./test/*; do
  echo "$eachfile"
  python predict.py -i "$eachfile" -o "./test/$((i++)).jpg"
done