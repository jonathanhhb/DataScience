#nvidia-docker run -it -w="/shared/DTK_Emulator/" -v /home/jbloedow/data/share_with_docker:/shared nvcr.io/idmod_idmtest2/idm_keras_emulator:0.4 ./dtk_emulator_itn.py 
nvidia-docker run -it -w="/shared/" -v $PWD:/shared nvcr.io/idmod/idm_keras_emulator:0.4 ./dtk_emulator_itn.py 
gnuplot -persist  -e "set datafile separator ','; plot '~/data/share_with_docker/DTK_Emulator/data.csv' smooth unique with linespoints; pause -1" 
