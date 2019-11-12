from data.data import TextMTData
from model import Encoder, Decoder, Attention, Seq2Seq
from config import DEBUG, device

import tqdm, time, random

import torch
from torch import nn
from torch import optim

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(model, n_iters, criterion, dataset, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    training_pairs = [dataset.tensorsFromPair(random.choice(dataset.pairs), device=device)
                      for i in range(n_iters)]

    input_tensor, target_tensor = zip(*training_pairs)
# def train(model, iterator, optimizer, criterion, clip):
    for i in range(20):
        batch_iterator = batch_data(input_tensor, target_tensor, batch_size=2048, device=device, pbar=tqdm.tqdm(position=0))
        loss = train(model, iterator=batch_iterator, optimizer=optimizer, criterion=criterion, loss_bar=tqdm.tqdm(position=1))

        # print("Loss", loss)

    criterion = nn.NLLLoss()
    # showPlot(plot_losses)

def train(model, iterator, optimizer, criterion, loss_bar=None):

    model.train()
    epoch_loss = 0
    samples = 0
    if loss_bar is not None:
        loss_bar.total = 10
    for i, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]
        samples += len(src)

        optimizer.zero_grad()
        output = model(src, trg)
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].contiguous().view(-1)

        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if loss_bar is not None:
            loss_bar.n = epoch_loss/samples
            loss_bar.write(str(epoch_loss/samples))
            # loss_bar.total = max([loss_bar.total, loss.item()])
            loss_bar.update(0)
        epoch_loss += loss.item()

    return epoch_loss / samples

def batch_data(input_data, target_data, batch_size, device, pbar=None):
    global PAD_IDX
    index = 0
    if pbar is not None:
        pbar.total = int(len(input_data)/batch_size)
    last = False
    while index < len(input_data)/batch_size:
        if (index+1)*batch_size < len(input_data):
            batch_input = input_data[index*batch_size: (index+1)*batch_size]
            batch_target = target_data[index*batch_size: (index+1)*batch_size]
        else:
            batch_input = input_data[index*batch_size:]
            batch_target = target_data[index*batch_size:]
            batch_size = len(batch_input)
            last = True
        input_lengths = [len(p) for p in batch_input]
        target_lengths = [len(p) for p in batch_target]
        padded_input = torch.nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=PAD_IDX).view(batch_size, -1)
        padded_target = torch.nn.utils.rnn.pad_sequence(batch_target, batch_first=True, padding_value=PAD_IDX).view(batch_size, -1)
        yield [padded_input.permute(1, 0), padded_target.permute(1, 0)]
        # yield [torch.nn.utils.rnn.pack_padded_sequence(padded_input, input_lengths, batch_first=True, enforce_sorted=False),
        # torch.nn.utils.rnn.pack_padded_sequence(padded_target, target_lengths, batch_first=True, enforce_sorted=False)]
        if pbar is not None:
            pbar.n = index
            pbar.update(0)
        index+=1
        if last:
            break

if __name__ == '__main__':
    global PAD_IDX

    tmt = TextMTData('eng', 'fra')
    input_lang, output_lang, pairs = tmt.prepareData(reverse=True)
    INPUT_DIM = input_lang.n_words
    OUTPUT_DIM = output_lang.n_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.init_weights()

    print(f'The model has {model.count_parameters():,} trainable parameters')


    optimizer = optim.Adam(model.parameters())

    PAD_IDX = 3
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    trainIters(model, 75000, criterion, dataset=tmt, print_every=5000)
