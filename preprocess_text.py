# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: text_preprocess.py
@Time: 2021/3/31 22:14
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:44:08 2020

@author: hx301
"""
# NOTE: Functions for preprocessing text data, removing special characters, converting text to lowercase, and tokenizing text.

# NOTE: functions:
# 1. `unicode_to_ascii`: Converts the unicode sentence to ascii (Normalize string in NFD form)
# 2. `normalize_string`: normalize characters, remove special characters, and convert text to lowercase
# 3. `cutted_data`: only use length of 4-30 words sentences
# 4. `save_clean_sentences`: save cleaned sentences to file in wb mode
# 5. `process`: use 2,3 to preprocess text data
# 6. `tokenize`: tokenize text data (<START> and <END> tokens are added to the sentences, convert text to tokens)
# 7. `build_vocab`: after tokenization, build the vocab and assign each token an index (exclude token that appears less than 2 times)
# 8. `encode`: encode token to index
# 9. `decode`: decode indices to tokens or sequences joined by delim (e.g., ' ')


import argparse
import json
import os
import pickle
import re  # work with regular expressions
import unicodedata

from tqdm import tqdm  # progress bar for loops
from w3lib.html import remove_tags  # useful for removing html tags

parser = argparse.ArgumentParser()
parser.add_argument("--input-data-dir", default="europarl/en", type=str)
parser.add_argument("--output-train-dir", default="europarl/train_data.pkl", type=str)
parser.add_argument("--output-test-dir", default="europarl/test_data.pkl", type=str)
parser.add_argument("--output-vocab", default="europarl/vocab.json", type=str)

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3,
}


def unicode_to_ascii(s):
    """Convert unicode to ascii in NFD form and remove diacritics"""
    return "".join(
        # if the unicode data is not mark, nonspacing characters then normaliz3e it to NFD form
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    """normalize string by removing html tags, covnerting to lowercase, and adding whilte space before !.?"""
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove the XML-tags
    s = remove_tags(s)
    # add white space before !.?
    s = re.sub(r"([!.?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    # change to lower letter
    s = s.lower()
    return s


def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    """
    ensure sentences are in length of 4 to 30 words length
    """
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(" ".join(line))
    return cutted_lines


def save_clean_sentences(sentence, save_path):
    """
    save the cleaned sentences to a file (save_path file) using pickle
    """
    pickle.dump(sentence, open(save_path, "wb"))  # wb is write binary mode
    print("Saved: %s" % save_path)


def process(text_path):
    """
    load the sentences from file and process the sentences into lceaned sentences
    """
    fop = open(text_path, "r", encoding="utf8")  # read mode
    raw_data = fop.read()
    sentences = raw_data.strip().split("\n")  #  split raw data by new line character
    raw_data_input = [
        normalize_string(data) for data in sentences
    ]  # normalize the strings in the sentences
    raw_data_input = cutted_data(raw_data_input)  # cut the cleaned_input
    fop.close()

    return raw_data_input


def tokenize(
    s,
    delim=" ",
    add_start_token=True,
    add_end_token=True,
    punct_to_keep=None,
    punct_to_remove=None,
):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, "%s%s" % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, "")

    tokens = s.split(delim)  # split the string by the delimiter
    if add_start_token:
        tokens.insert(0, "<START>")
    if add_end_token:
        tokens.append("<END>")
    return tokens


def build_vocab(
    sequences,
    token_to_idx={},
    min_token_count=1,
    delim=" ",
    punct_to_keep=None,
    punct_to_remove=None,
):
    """
    build vocabulary for the sequences (dict of token to index mapping)
    """
    token_to_count = {}

    for seq in sequences:
        seq_tokens = tokenize(
            seq,
            delim=delim,
            punct_to_keep=punct_to_keep,
            punct_to_remove=punct_to_remove,
            add_start_token=False,
            add_end_token=False,
        )
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    """
    encode a sequence of tokens to a sequence of indices usign the vocab provided
    """
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = "<UNK>"
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    """
    decode a sequence of indices to a sequence of tokens using teh vocab
    """
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == "<END>":
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)

# NOTE: main function
# - read data from file and process it into sentences
# - remove same sentences
# - index-based encoding
#   - build vocab (build token to indices dict)
#   - save vocab to file
#   - encode txt to indices
# - split encoded data into train and test sets (90/10 split)
# - write data to files using pickle

def main(args):
    data_dir = "data/"
    args.input_data_dir = data_dir + args.input_data_dir
    args.output_train_dir = data_dir + args.output_train_dir
    args.output_test_dir = data_dir + args.output_test_dir
    args.output_vocab = data_dir + args.output_vocab

    print(args.input_data_dir)
    sentences = []
    print("Preprocess Raw Text")
    for fn in tqdm(os.listdir(args.input_data_dir)):
        if not fn.endswith(".txt"):
            continue
        # process, clean setnences
        process_sentences = process(os.path.join(args.input_data_dir, fn))
        sentences += process_sentences

    # remove the same sentences
    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1  # ISSUE: meaningless if we dont use the sentence count?
    sentences = list(a.keys())
    print("Number of sentences: {}".format(len(sentences)))

    print("Build Vocab")
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS, punct_to_keep=[";", ","], punct_to_remove=["?", "."]
    )

    vocab = {"token_to_idx": token_to_idx}
    print("Number of words in Vocab: {}".format(len(token_to_idx)))

    # save the vocab
    if args.output_vocab != "":
        with open(args.output_vocab, "w") as f:
            json.dump(vocab, f)

    print("Start encoding txt")
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[";", ","], punct_to_remove=["?", "."])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    print("Writing Data")
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9) :]

    with open(args.output_train_dir, "wb") as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
