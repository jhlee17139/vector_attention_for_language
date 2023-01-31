import torch
from tools.word_tools import generate_square_subsequent_mask, create_mask
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtext.data.metrics import bleu_score


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_idx, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform, src_language, tgt_language,
              bos_idx, eos_idx, device):
    model.eval()
    src = text_transform[src_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=bos_idx, eos_idx=eos_idx, device=device).flatten()
    sentence = " ".join(vocab_transform[tgt_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    sentence = sentence[1:].rstrip(' ')
    sentence = sentence[:-2] + sentence[-1:]

    return sentence


def blue(output_list, target_list):
    blue_output_list = []
    blue_target_list = []

    # split sentence
    for i in range(len(output_list)):
        output = output_list[i]
        target = target_list[i]

        output = output.split(' ')
        target = target.split(' ')

        blue_output_list.append(output)
        blue_target_list.append([target])

    max_len = 4
    weight_scalar = 1.0 / max_len
    score = bleu_score(blue_output_list, blue_target_list, max_n=max_len, weights=[weight_scalar, ] * max_len)

    return score
