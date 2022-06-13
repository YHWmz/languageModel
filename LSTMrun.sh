#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

python main.py --data data/gigaspeech --nlayers 3 --emsize 600 --nhid 600 --batch_size 20 --lr 15 --model LSTM --cuda --tied --save ./modelsave/LSTM_drop01.pt --dropout 0.3 --clip 0.2 --tbsave tbsave/LSTM_best

#python main.py --data data/gigaspeech --nlayers 1 --emsize 600 --nhid 600 --batch_size 20 --lr 15 --model LSTM --cuda --tied --save ./modelsave/LSTM_default.pt

#python main.py --data data/gigaspeech --emsize 600 --nhid 300 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 600 --nhid 400 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 600 --nhid 500 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 600 --nhid 700 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 600 --nhid 800 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt

#python main.py --data data/gigaspeech --emsize 100 --nhid 100 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 200 --nhid 200 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 300 --nhid 300 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 400 --nhid 400 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 500 --nhid 500 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 600 --nhid 600 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 700 --nhid 700 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --emsize 800 --nhid 800 --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt

#python main.py --data data/gigaspeech --batch_size 1 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 5 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 10 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 40 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 80 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 160 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 320 --lr 15 --model LSTM --cuda --save ./modelsave/LSTM_default.pt

### lr实验
#python main.py --data data/gigaspeech --batch_size 20 --lr 1 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 3 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 5 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 7 --model LSTM --cuda --save ./modelsave/LSTM_default.pt
#python main.py --data data/gigaspeech --batch_size 20 --lr 10 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 15 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 18 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 22 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 25 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 20 --lr 30 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 10 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 15 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 18 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 22 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 25 --lr 20 --model LSTM --cuda
#python main.py --data data/gigaspeech --batch_size 30 --lr 20 --model LSTM --cuda








