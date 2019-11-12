from io import open
import unicodedata
import string
import re
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
"i am ", "i m ",
"he is", "he s ",
"she is", "she s ",
"you are", "you re ",
"we are", "we re ",
"they are", "they re "
)

######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence, device, length=None):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        if length:
            indexes = indexes + [0]*(length - len(indexes))
            return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

        return torch.tensor(indexes, device=device)

# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))
class TextMTData(object):
    def __init__(self, lang1, lang2):
        self.input_lang_name = lang1
        self.output_lang_name = lang2
        self.input_lang, self.output_lang, self.pairs = None, None, None
        # self.prepareData(reverse=True)

    ######################################################################
    # To read the data file we will split the file into lines, and then split
    # lines into pairs. The files are all English → Other Language, so if we
    # want to translate from Other Language → English I added the ``reverse``
    # flag to reverse the pairs.
    #
    def readLangs(self, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('data/%s-%s.txt' % (self.input_lang_name, self.output_lang_name), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        self.pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            self.pairs = [list(reversed(p)) for p in self.pairs]
            self.input_lang = Lang(self.output_lang_name)
            self.output_lang = Lang(self.input_lang_name)
        else:
            self.input_lang = Lang(self.input_lang_name)
            self.output_lang = Lang(self.output_lang_name)

        # return input_lang, output_lang, pairs

    ######################################################################
    # Since there are a *lot* of example sentences and we want to train
    # something quickly, we'll trim the data set to only relatively short and
    # simple sentences. Here the maximum length is 10 words (that includes
    # ending punctuation) and we're filtering to sentences that translate to
    # the form "I am" or "He is" etc. (accounting for apostrophes replaced
    # earlier).
    #
    def filterPair(self, p):
        # print(p, eng_prefixes, MAX_LENGTH)
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)


    def filterPairs(self):
        # print(len(self.pairs))
        x = [pair for pair in self.pairs if self.filterPair(pair)]
        # print(x, len(x))
        return x

    ######################################################################
    # The full process for preparing the data is:
    #
    # -  Read text file and split into lines, split lines into pairs
    # -  Normalize text, filter by length and content
    # -  Make word lists from sentences in pairs
    #

    def prepareData(self, reverse=False):
        self.readLangs(reverse)
        print("Read %s sentence pairs" % len(self.pairs))
        self.pairs = self.filterPairs()
        print("Trimmed to %s sentence pairs" % len(self.pairs))
        print("Counting words...")
        for pair in self.pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])
        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        return self.input_lang, self.output_lang, self.pairs

    def tensorsFromPair(self, pair, device, length=None):
        input_tensor = self.input_lang.tensorFromSentence(pair[0], device, length)
        target_tensor = self.output_lang.tensorFromSentence(pair[1], device, length)
        return (input_tensor, target_tensor)



if __name__ == '__main__':
    tmt = TextMTData('eng', 'fra')
    tmt.prepareData(reverse=True)
