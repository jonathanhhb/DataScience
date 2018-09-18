#!/bin/bash
nrunx -v ~/GIT/DataScience/DTKEmulator/MALARIA:/workspace nvcr.io/idmod/keras:tensor17.10 ./dtk_emulator_itn.py
printf '%s\n' "set datafile separator ','" "plot '~/GIT/DataScience/DTKEmulator/MALARIA/data.csv' smooth unique with linespoints" "pause -1" "" > plotIt
nrunx -v ~/GIT/DataScience/DTKEmulator/MALARIA:/plot nvcr.io/idmod/gnuplot:ubnuntu18.04_1 "-p" /plot/plotIt
