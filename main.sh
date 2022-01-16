#! /bin/bash
python system_cnn.py -channel_mode awgn -modem_num 4 -snr 5 -lr 1e-3
python system_main.py -channel_mode awgn -curve cnn -modem_num 4 -snr_start -1 -snr_start 3 -snr_step 1
python system_main.py -channel_mode awgn -curve unquantized -modem_num 4 -snr_start -1 -snr_start 3 -snr_step 1
python system_main.py -channel_mode awgn -curve quantized -modem_num 4 -snr_start -1 -snr_start 3 -snr_step 1
python system_cnn.py -channel_mode awgn -modem_num 16 -snr 10 -lr 1e-4
python system_main.py -channel_mode awgn -curve cnn -modem_num 16 -snr_start -2 -snr_start 13 -snr_step 1
python system_main.py -channel_mode awgn -curve unquantized -modem_num 16 -snr_start -2 -snr_start 13 -snr_step 1
python system_main.py -channel_mode awgn -curve quantized -modem_num 16 -snr_start -2 -snr_start 13 -snr_step 1
python system_cnn.py -channel_mode fading -modem_num 16 -snr 10 -lr 1e-4
python system_main.py -channel_mode fading -curve cnn -modem_num 16 -snr_start -2 -snr_start 13 -snr_step 1
python system_main.py -channel_mode fading -curve unquantized -modem_num 16 -snr_start -2 -snr_start 13 -snr_step 1
python system_main.py -channel_mode fading -curve quantized -modem_num 16 -snr_start -2 -snr_start 13 -snr_step 1 