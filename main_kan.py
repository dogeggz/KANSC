# -*- coding: utf-8 -*-

"""
Created on Tue May 26 16:59:14 2020
Modified on Wed Jun 12 2024

@modified by: Zifan Zhu
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
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_data
from models.kansc import KANSC
from models.mutual_info import Mine
from utils import SNR_to_noise, initNetParams, train_mi, train_step, val_step

"""
parser used for command line arguments include vocab_file(stores the vocabulary that maps tokens to indices or vice versa), checkpoint_path(stores the checkpoints for the model), channel(defines the channel type for channel processing), max_length(maximum length of the sentence), min_length(minimum length of the sentence), d_model(dimension of the model), dk(dimension of the KAN network), num_layers(number of layers in the model), num_heads(number of heads in the model), batch_size(batch size for training), epochs(number of epochs for training)
"""
parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument("--vocab-file", default="europarl/vocab.json", type=str)
parser.add_argument("--checkpoint-path", default="checkpoints/kansc-AWGN", type=str)
parser.add_argument(
    "--channel",
    default="AWGN",
    type=str,
    help="Please choose AWGN, Rayleigh, and Rician",
)
parser.add_argument("--MAX-LENGTH", default=30, type=int)
parser.add_argument("--MIN-LENGTH", default=4, type=int)
parser.add_argument("--d-model", default=128, type=int)
# NOTE: added parameter dk as the hidden layer size for KAN
parser.add_argument("--dks", default=64, type=int)
parser.add_argument("--dkc", default=256, type=int)
parser.add_argument("--num-layers", default=3, type=int)
parser.add_argument("--num-heads", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--epochs", default=80, type=int)
parser.add_argument(
    "--restart-epochs",
    default=40,
    type=int,
    help="Number of epochs for learning rate scheduler restart",
)
# cosineannealing
parser.add_argument("--lr", default=0.0002, type=float, help="start learning rate")
parser.add_argument("--lrf", default=0.00005, type=float, help="end learning rate")
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--test-mode", default=False, type=bool)


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
                "Epoch: {}; Type: VAL;   Loss: {:.5f}".format(epoch + 1, loss)
            )

    # average loss for the test dataset as the validation loss
    return total / len(test_iterator)


# train function for training the model
def train(epoch, args, net, mi_net=None):
    # train dataset
    train_eur = EurDataset("train")
    # train dataset data foader with batch size defined in args
    train_iterator = DataLoader(
        train_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data,
    )
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    train_total = 0
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
            pbar.set_description(
                "Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}".format(
                    epoch + 1, loss, mi
                )
            )
            train_total += loss
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
            pbar.set_description(
                "Epoch: {}; Type: Train; Loss: {:.5f}".format(epoch + 1, loss)
            )
            train_total += loss
    train_total /= len(train_iterator)
    return train_total


if __name__ == "__main__":
    seed = 1
    setup_seed(seed)
    args = parser.parse_args()
    test_mode = args.test_mode
    test_checkpoint_path = "checkpoints/test" + (
        "-awgn" if args.channel == "AWGN" else "-rayleigh"
    )
    print("Seed", seed)
    print("Channel:", args.channel)
    print("Semantic KAN hidden layer size: ", args.dks)
    print("Channel KAN hidden layer size: ", args.dkc)
    print("batch size:", args.batch_size)
    print("test mode", test_mode)

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
    kansc = KANSC(
        args.num_layers,
        num_vocab,
        num_vocab,
        num_vocab,
        num_vocab,
        args.d_model,
        args.num_heads,
        args.dks,
        args.dkc,
        args.dropout,
    ).to(device)
    # mutual information net
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    # adam optimizer for kansc
    optimizer = torch.optim.Adam(kansc.parameters(), lr=args.lr)

    # NOTE: we add a learning rate scheduler to adjust the learning rate
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_epochs, T_mult=1, eta_min=args.lrf
    )
    print("Number of epochs to restart: ", args.restart_epochs)
    print("Start learning rate: ", args.lr)
    print("End learning rate: ", args.lrf)

    # optimizer for mutual information
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    initNetParams(kansc)
    val_loss_record = []
    train_loss_record = []
    learning_rate_record = []
    for epoch in range(args.epochs):
        start = time.time()
        record_loss = 10

        train_loss = train(epoch, args, kansc)
        scheduler.step(epoch + 1)
        train_loss_record.append(train_loss)
        last_lr = scheduler.get_last_lr()[0]
        learning_rate_record.append(last_lr)
        val_loss = validate(epoch, args, kansc)
        val_loss_record.append(val_loss)
        print("----Train: ", train_loss)
        print("----Val:   ", val_loss)
        print("----LR:    ", last_lr)

        if val_loss < record_loss:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(
                (args.checkpoint_path if not test_mode else test_checkpoint_path)
                + "/checkpoint_{}.pth".format(str(epoch + 1).zfill(2)),
                "wb",
            ) as f:
                torch.save(kansc.state_dict(), f)
            record_loss = val_loss

    # Record losses and learning rate
    # print("Val Loss:", val_loss_record)
    # val_file_name = "val_loss_" + args.channel + ".txt"
    # record_path = 'kansc' if not test_mode else 'test'
    # f = open(f"record/{record_path}/" + val_file_name, "w")
    # s = str(val_loss_record)
    # f.write(s)
    # f.close()
    # train_file_name = "train_loss_" + args.channel + ".txt"
    # f = open(f"record/{record_path}/" + train_file_name, "w")
    # s = str(train_loss_record)
    # f.write(s)
    # f.close()
    # lr_file_name = "lr_" + args.channel + ".txt"
    # f = open(f"record/{record_path}/" + lr_file_name, "w")
    # s = str(learning_rate_record)
    # f.write(s)
    # f.close()
