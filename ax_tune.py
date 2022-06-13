# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from torch.utils.tensorboard import SummaryWriter
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN
import data
import model

def train_evaluate(parameterization):
    # Set the random seed manually for reproducibility.

    # {"name": "learning_rate", "type": "choice", "values": [13, 14, 15, 16, 17]},
    # {"name": "emsize", "type": "choice", "values": [500, 600, 700, 800]},
    # {"name": "numlayer", "type": "choice", "values": [2, 3, 4, 5]},
    # {"name": "clip", "type": "range", "bounds": [0.1, 0.6], "value_type": "float"},
    # {"name": "dropout", "type": "range", "bounds": [0, 0.5], "value_type": "float"},

    args.lr = parameterization.get("learning_rate", 1)
    args.emsize = parameterization.get("emsize", 600)
    args.nhid = parameterization.get("nhid", 700)
    args.numlayer = parameterization.get("numlayer", 4)  # 时间步长
    args.clip = parameterization.get("clip", 0.25)
    args.dropout = parameterization.get("dropout", 0.2)
    args.nhead = parameterization.get("numhead", 2)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)
    batch_num = train_data.size(0) // args.bptt

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        Model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        Model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    criterion = nn.NLLLoss()

    print ("Vocabulary Size: ", ntokens)
    num_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print ("Total number of model parameters: {:.2f}M".format(num_params*1.0/1e6))

    ###############################################################################
    # Training code
    ###############################################################################

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target


    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        Model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        if args.model != 'Transformer':
            hidden = Model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i)
                if args.model == 'Transformer':
                    output = Model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = Model(data, hidden)
                    hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)


    def train(epoch):
        # Turn on training mode which enables dropout.
        Model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        all_loss = 0
        loss_cnt = 0
        if args.model != 'Transformer':
            hidden = Model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            Model.zero_grad()
            if args.model == 'Transformer':
                output = Model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = Model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(Model.parameters(), args.clip)
            for p in Model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time

                all_loss += cur_loss
                loss_cnt += 1
                # print(cur_loss, batch//args.log_interval//2 + epoch * batch//args.log_interval//2)
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            if args.dry_run:
                break

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(Model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        if lr <= 0.01:
            break

    with open(args.save, 'rb') as f:
        Model = torch.load(f)
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            Model.rnn.flatten_parameters()

    test_loss = evaluate(test_data)
    return -test_loss




parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='data/gigaspeech',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=7,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "learning_rate", "type": "range", "bounds": [0.1, 2], "value_type": "float"},
        {"name": "emsize", "type": "choice", "values": [500,600,700,800]},
        {"name": "nhid", "type": "choice", "values": [500,600,700,800]},
        {"name": "numlayer", "type": "choice", "values": [2,3,4,5]},
        {"name": "numhead", "type": "choice", "values": [2,3,4,5]},
        {"name": "clip", "type": "range", "bounds": [0.1, 0.6], "value_type": "float"},
        {"name": "dropout", "type": "range", "bounds": [0, 0.5], "value_type": "float"},

    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)
print(best_parameters)
with open('tmp.txt', 'w') as f:
    f.write(best_parameters)
