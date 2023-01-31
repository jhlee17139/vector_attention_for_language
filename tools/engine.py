from torch.utils.data import DataLoader
from tools.word_tools import create_mask
from tqdm import tqdm


def train_epoch(model, optimizer, loss_fn, batch_size, device, dataset, collate_fn):
    model.train()
    losses = 0
    train_iter = dataset
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    num_iter = 0

    for _, (src, tgt) in enumerate(tqdm(train_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        num_iter += 1

    return losses / num_iter


def evaluate(model, loss_fn, batch_size, device, dataset, collate_fn):
    model.eval()
    losses = 0
    num_iter = 0

    val_iter = dataset
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

    for _, (src, tgt) in enumerate(tqdm(val_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        num_iter += 1

    return losses / num_iter

