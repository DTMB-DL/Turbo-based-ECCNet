#! /bin/bash 
#python system_qpsk2_cnn.py -channel_mode awgn -modem_num 16 -snr 15 -lr_encoder 5e-4 -lr_decoder 5e-4
#python system_cnn.py -channel_mode awgn -modem_num 16 -snr 15
#python system_main.py -channel_mode awgn -modem_num 16
python system_qpsk2_cnn.py -channel_mode awgn -modem_num 4 -snr 5 -lr_encoder 5e-4 -lr_decoder 5e-4
python system_main.py -channel_mode awgn -modem_num 4 -snr_start -1 -snr_end 3 -snr_step 0.25
 
