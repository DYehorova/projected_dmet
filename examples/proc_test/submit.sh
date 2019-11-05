#!/bin/bash

for file in nproc*
do

echo $file

cd $file

qsub job_run

cd ..

done

