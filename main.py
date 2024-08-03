# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import argparse  # argparse is a module that provides a command line interface
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_data
from models.mutual_info import Mine
from models.transceiver import DeepSC
from utils import SNR_to_noise, initNetParams, train_mi, train_step, val_step

"""
parser used for command line arguments include vocab_file(stores the vocabulary that maps tokens to indices or vice versa), checkpoint_path(stores the checkpoints for the model), channel(defines the channel type for channel processing), max_length(maximum length of the sentence), min_length(minimum length of the sentence), d_model(dimension of the model), dff(dimension of the feed forward network), num_layers(number of layers in the model), num_heads(number of heads in the model), batch_size(batch size for training), epochs(number of epochs for training)
"""
parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument("--vocab-file", default="europarl/vocab.json", type=str)
parser.add_argument(
    "--checkpoint-path", default="checkpoints/deepsc-Rayleigh", type=str
)
parser.add_argument(
    "--channel",
    default="Rayleigh",
    type=str,
    help="Please choose AWGN, Rayleigh, and Rician",
)
parser.add_argument("--MAX-LENGTH", default=30, type=int)
parser.add_argument("--MIN-LENGTH", default=4, type=int)
parser.add_argument("--d-model", default=128, type=int)
parser.add_argument("--dff", default=256, type=int)
parser.add_argument("--num-layers", default=3, type=int)
parser.add_argument("--num-heads", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--epochs", default=80, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print("CUDA Available: ", torch.cuda.is_available())


# setup teh seed for all the randome number generators
def setup_seed(seed):
    # torch randome nubmer gnerator
    torch.manual_seed(seed)
    # cuda random number generator
    torch.cuda.manual_seed_all(seed)
    # numpy random number generator
    np.random.seed(seed)
    # random random number generator
    random.seed(seed)
    # NOTE: set this may lead to slower training time, sacrifice some computational efficiency but gain consistency in the results
    torch.backends.cudnn.deterministic = True


# validate function for model performance (loss on test dataset)
def validate(epoch, args, net):
    # load test dataset
    test_eur = EurDataset("test")

    # dataloadder for test data with batch size defined in args
    # collate_data function is used to pad the sentenc3es with zeros as collate_fn in dataloader
    test_iterator = DataLoader(
        test_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data,
    )
    net.eval()  # model to evaluation mode
    pbar = tqdm(test_iterator)  # progress bar for test dataloader
    total = 0  # total loss
    with torch.no_grad():  # ensures that the model is not trained ( turn off gradient descent )
        for sents in pbar:  # iterate throught the test data
            sents = sents.to(device)  # move to GPU/cpu for calculation
            loss = val_step(
                net, sents, sents, 0.1, pad_idx, criterion, args.channel
            )  # calculate the loss for this batch of test data

            total += loss  # accumulate the loss
            pbar.set_description(
                "Epoch: {}; Type: VAL; Loss: {:.5f}".format(epoch + 1, loss)
            )

    # average loss for the test dataset as the validation loss
    return total / len(test_iterator)


# train function for training the model
def train(epoch, args, net, mi_net=None):
    # train dataset
    train_eur = EurDataset("train")
    # train dataset data loader with batch size defined in args
    train_iterator = DataLoader(
        train_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data,
    )
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))
    train_loss = 0

    # for train data batches
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # we need to train mutual information model
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            # loss from training the model
            loss = train_step(
                net,
                sents,
                sents,
                0.1,
                pad_idx,
                optimizer,
                criterion,
                args.channel,
                mi_net,
            )
            train_loss += loss
            pbar.set_description(
                "Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}".format(
                    epoch + 1, loss, mi
                )
            )
        else:
            # train the whole network
            loss = train_step(
                net,
                sents,
                sents,
                noise_std[0],
                pad_idx,
                optimizer,
                criterion,
                args.channel,
            )
            train_loss += loss
            pbar.set_description(
                "Epoch: {}; Type: Train;   Loss: {:.5f}".format(epoch + 1, loss)
            )
    return train_loss / len(train_iterator)


if __name__ == "__main__":
    setup_seed(10)
    args = parser.parse_args()
    print("Channel:", args.channel)
    args.vocab_file = "data/" + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, "rb"))
    # token to idx mapping
    token_to_idx = vocab["token_to_idx"]
    num_vocab = len(token_to_idx)
    # indices for padding , start, and end(special tokens)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(
        args.num_layers,
        num_vocab,
        num_vocab,
        num_vocab,
        num_vocab,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1,
    ).to(device)
    # mutual information net
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    # use adam optimizer for training

    # NOTE: Adam optimizer
    """
     NOTE 1. Adaptive learning rate, adjusts lr for each parameter based on their gradient magnitudes, improving efficiency for parameters with different behaviors 
     NOTE 2. Efficiency. handles non-stationary objectives and noisy or sparse gradients well
     NOTE 3. Little memory requirement, maintains moderate computataional overhaed by storign only two vectors representing gradients first and second moments
     NOTE 4. Robustness. performs well across various settings due to its adaptive approach to adjusting lra, making it less sensitive to inital settings
     NOTE 5. Well suited for large datasets or parameters, 
     NOTE 6. Ease of implementation, requires minimal configuration and providing reliable performance with its default settings in diverse tasks
    """
    optimizer = torch.optim.Adam(
        deepsc.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4
    )
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)
    train_loss_record = []
    val_loss_record = []
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10  # ISSUE: record_acc is actually record_loss

        train_loss = train(epoch, args, deepsc)
        train_loss_record.append(train_loss)
        val_loss = validate(epoch, args, deepsc)
        val_loss_record.append(val_loss)
        print("----", val_loss)

        if val_loss < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(
                args.checkpoint_path
                + "/checkpoint_{}.pth".format(str(epoch + 1).zfill(2)),
                "wb",
            ) as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = val_loss

    print("Train Loss:", train_loss_record)
    print("Val Loss:", val_loss_record)
    val_file_name = "val_loss_" + args.channel + ".txt"
    f = open("record/deepsc/" + val_file_name, "w")
    s = str(val_loss_record)
    f.write(s)
    f.close()
    train_file_name = "train_loss_" + args.channel + ".txt"
    f = open("record/deepsc/" + train_file_name, "w")
    s = str(train_loss_record)
    f.write(s)
    f.close()
