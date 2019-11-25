from data.data import MUSTCData as Dataset
from model import Encoder, Decoder, Attention, Seq2Seq, WavEncoder
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

def trainIters(model, n_iters, criterion, dataset, print_every=1000, plot_every=100, learning_rate=0.01, input_lang=None, output_lang=None):
    start = time.time()
    def get_pairs():
        for i in range(n_iters):
            yield dataset.tensorsFromPair(random.choice(dataset.pairs), device=device)
    print("Getting pairs")
                      # for i in range(n_iters)]

    print("Separating pairs")
    # input_tensor, target_tensor = zip(*training_pairs)
# def train(model, iterator, optimizer, criterion, clip):
    print("Starting")
    for i in tqdm.trange(20000):
        training_pairs = get_pairs()# [dataset.tensorsFromPair(random.choice(dataset.pairs), device=device)
        batch_iterator = batch_data(training_pairs, batch_size=64, device=device, pbar=tqdm.tqdm(position=1))
        loss = train(model, iterator=batch_iterator, optimizer=optimizer, criterion=criterion, loss_bar=tqdm.tqdm(position=2),
            input_lang=input_lang, output_lang=output_lang)

        # print("Loss", loss)

    criterion = nn.NLLLoss()
    # showPlot(plot_losses)

def train(model, iterator, optimizer, criterion, loss_bar=None, input_lang=None, output_lang=None):

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
        output = model(src, trg, 0.9)
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output.view(-1, output.shape[-1])
        trg = trg.contiguous().view(-1)

        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # print(batch[1], batch[1].size())

        test_input = torch.stack([src.permute(1, 0, 2)[0]]).permute(1, 0, 2)
        target_input = torch.stack([batch[1].permute(1, 0)[0]]).permute(1, 0)
        test_output = model(test_input, torch.tensor([[trg[0].item()]], device=device), 0, True)

        test_indeces = test_input.view(-1)
        input_sentence = input_lang.get_sentence(test_indeces)

        target_indeces = target_input.view(-1)
        target_sentence = output_lang.get_sentence(target_indeces)

        # print(test_output, test_output.size())
        output_indeces = test_output.argmax(2)
        output_sentence = output_lang.get_sentence(output_indeces)

        if loss_bar is not None:
            loss_bar.n = epoch_loss/samples
            loss_bar.write(str(loss))
            loss_bar.write(input_sentence)
            loss_bar.write(target_sentence)
            loss_bar.write(output_sentence)
            # loss_bar.total = max([loss_bar.total, loss.item()])
            loss_bar.update(0)
        else:
            print()
        epoch_loss += loss.item()

    return epoch_loss / samples

def batch_data(data, batch_size, device, pbar=None):
    global PAD_IDX
    index = 0
    if pbar is not None:
        try:
            pbar.total = int(len(data)/batch_size)
        except:
            pass
    last = False
    while True: #index < len(data)/batch_size:
        cur_index = 0
        batch_input = []
        batch_target = []
        while cur_index < batch_size:
            try:
                new_input, new_output = next(data)
                batch_input.append(new_input)
                batch_target.append(new_output)
                cur_index += 1
            except StopIteration:
                last = True
                break

        cur_batch_size = len(batch_input)
        # if (index+1)*batch_size < len(input_data):
        #     batch_input = input_data[index*batch_size: (index+1)*batch_size]
        #     batch_target = target_data[index*batch_size: (index+1)*batch_size]
        # else:
        #     batch_input = input_data[index*batch_size:]
        #     batch_target = target_data[index*batch_size:]
        #     batch_size = len(batch_input)
        #     last = True
        input_lengths = [len(p) for p in batch_input]
        target_lengths = [len(p) for p in batch_target]
        padded_input = torch.nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=PAD_IDX).view(cur_batch_size, -1, 40)
        padded_target = torch.nn.utils.rnn.pad_sequence(batch_target, batch_first=True, padding_value=PAD_IDX).view(cur_batch_size, -1)
        yield [padded_input.permute(1, 0, 2), padded_target.permute(1, 0)]
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

    tmt = Dataset('en', 'de')
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
    enc = WavEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.init_weights()

    print(f'The model has {model.count_parameters():,} trainable parameters')


    optimizer = optim.Adam(model.parameters())

    PAD_IDX = 2
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    trainIters(model, 60, criterion, dataset=tmt, print_every=5000, input_lang=input_lang, output_lang=output_lang)
