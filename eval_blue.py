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
from tools.eval import translate, blue
from torch.nn.utils.rnn import pad_sequence
from model.seq2seq import Seq2SeqTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm


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
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    transformer.load_state_dict(torch.load(args.load_path))
    transformer = transformer.to(DEVICE)
    transformer.eval()

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

    print("translate de->en")
    test_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    test_dataloader = DataLoader(test_iter, batch_size=1)

    de_list = []
    en_list = []
    output_list = []

    for idx, test_data in tqdm(enumerate(test_dataloader)):
        de = test_data[0][0]
        en = test_data[1][0]
        de_to_en = translate(transformer, de, text_transform, vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, BOS_IDX,
                             EOS_IDX, DEVICE)

        de_list.append(de)
        en_list.append(en)
        output_list.append(de_to_en)

        if idx % 10 == 0:
            print()
            print('data {}'.format(idx))
            print('source : {}'.format(de_list[idx]))
            print('target : {}'.format(en_list[idx]))
            print('output : {}'.format(output_list[idx]))

    score = blue(output_list, en_list)
    print('blue score : {}'.format(score))


def get_args_parser():
    parser = argparse.ArgumentParser('transformer', add_help=False)
    parser.add_argument('--load_path', default='./output/dot_prod_model.pth', type=str)
    parser.add_argument('--attn_type', type=str, required=False, choices=['dot_prod', 'vec_prod'], default='dot_prod')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
