#!/bin/bash

num_jobs=5
prev_job_id=""

for ((i=1; i<=num_jobs; i++)); do
  if [ -z "$prev_job_id" ]; then
    # Submit the first job without any dependency
    output=$(sbatch --array=18 ./projects/lsst_y1/EXAMPLE_PolyChordX_SBU5.sbatch)
  else
    # Submit the subsequent jobs with a dependency on the previous job
    output=$(sbatch --dependency=afterany:$prev_job_id --array=18 ./projects/lsst_y1/EXAMPLE_PolyChordX_SBU5.sbatch)
  fi

  # Extract the job ID of the submitted job from the output
  prev_job_id=$(echo "$output" | awk '{print $4}')
  echo "Submitted job $i with ID: $prev_job_id"
done