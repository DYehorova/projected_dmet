#!/bin/bash

for file in nproc*
do

echo $file

cd $file

cp ../job_run .
cp ../dynamics_test.py .

nproc=`echo $file | sed -e 's/nproc//g'`

sed -i "s/MKL_NUM_THREADS=1/MKL_NUM_THREADS=$nproc/g" job_run
sed -i "s/ppn=1/ppn=$nproc/g" job_run

cd ..

done

