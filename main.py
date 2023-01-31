'''
https://pytorch.org/tutorials/beginner/translation_transformer.html
'''
import os
import argparse
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from timeit import default_timer as timer
from tools.word_tools import yield_tokens, sequential_transforms, tensor_transform
from tools.word_tools import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, special_symbols
from tools.engine import train_epoch, evaluate
from torch.nn.utils.rnn import pad_sequence
from model.seq2seq import Seq2SeqTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    multi30k.URL[
        "train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL[
        "valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'

    # Place-holders
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln, SRC_LANGUAGE, TGT_LANGUAGE,
                                                                     token_transform),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,
                                     attn_type=args.attn_type)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)  # Add BOS/EOS and create

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    NUM_EPOCHS = 18
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn, BATCH_SIZE, DEVICE, train_iter, collate_fn)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn, BATCH_SIZE, DEVICE, val_iter, collate_fn)
        print((f"\nEpoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    os.makedirs('./output', exist_ok=True)
    torch.save(transformer.state_dict(), '{}'.format(args.save_path))


def get_args_parser():
    parser = argparse.ArgumentParser('transformer', add_help=False)
    parser.add_argument('--save_path', default='./output/dot_prod_model.pth', type=str)
    parser.add_argument('--attn_type', type=str, required=False, choices=['dot_prod', 'vec_prod'], default='dot_prod')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

