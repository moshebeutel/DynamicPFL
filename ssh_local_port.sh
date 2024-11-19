#!/bin/bash


if [ -z "$1" ]; then
  echo "No argument provided! Map to dsief08"
  DSIEF=8
fi

# Store the argument in a variable
DSIEF="$1"

echo "Map port 600${DSIEF} to dsief0${DSIEF}:22"

ssh -L "600${DSIEF}:dsief0${DSIEF}:22" beutelm@dsihead.lnx.biu.ac.il