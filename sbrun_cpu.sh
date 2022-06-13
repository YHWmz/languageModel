#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./log/%j.out
#SBATCH --error=./log/%j.err

python main.py --data data/gigaspeech --epochs 6 --dropout 0.2 --save modelsave/RNN_TANH_default.pt --tbsave tbsave/RNN_TANH_default --model RNN_TANH
#python main.py --data data/gigaspeech --epochs 6 --dropout 0.2 --save modelsave/RNN_RELU_default.pt --tbsave tbsave/RNN_RELU_default --model RNN_RELU
#python main.py --data data/gigaspeech --epochs 6 --dropout 0.2 --save modelsave/LSTM_default.pt --tbsave tbsave/LSTM_default --model LSTM
#python main.py --data data/gigaspeech --epochs 6 --dropout 0.2 --save modelsave/GRU_default.pt --tbsave tbsave/GRU_default --model GRU
#python main.py --data data/gigaspeech --epochs 6 --dropout 0.2 --save modelsave/Transformer_default.pt --tbsave tbsave/Transformer_default --model Transformer

#python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
#python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
#python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
#python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
