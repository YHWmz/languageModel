#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

python main.py --data data/gigaspeech --nlayers 3 --emsize 600 --nhid 700 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH_drop01.pt --dropout 0.1 --tbsave tbsave/RNN_TANH_best

#python main.py --data data/gigaspeech --emsize 600 --nhid 700 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt --nlayers 1

#python main.py --data data/gigaspeech --emsize 100 --nhid 100 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 200 --nhid 200 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 300 --nhid 300 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 400 --nhid 400 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 500 --nhid 500 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 600 --nhid 600 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 700 --nhid 700 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 800 --nhid 800 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 900 --nhid 900 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --emsize 1000 --nhid 1000 --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt

#python main.py --data data/gigaspeech --batch_size 320 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 160 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 80 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 40 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 10 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 5 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 1 --lr 1 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt


#python main.py --data data/gigaspeech --batch_size 20 --lr 0.2 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 0.4 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 0.6 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 0.8 --model RNN_TANH --cuda --save ./modelsave/RNN_TANH.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 1 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 2 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 3 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 4 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 5 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 6 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 7 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 8 --model RNN_TANH --cuda
#python main.py --data data/gigaspeech --batch_size 10 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 15 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 18 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 22 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 25 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 30 --lr 20 --model LSTM --cuda








