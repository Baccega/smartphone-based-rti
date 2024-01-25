#!/bin/bash

for folder in "results/"*/
do 
  # Check if 'analysis_output.txt' exists in the folder
  if [[ -f "${folder}analysis_output.txt" ]]; then
    # Run the command on 'analysis_output.txt' within the folder
    echo "Processing ${folder}analysis_output.txt..."
    cat "${folder}analysis_output.txt" | grep SSIM
  else
    echo "File not found in ${folder}"
  fi
done