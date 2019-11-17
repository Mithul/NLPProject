import fasttext
import tqdm
from polyglot.text import Text
import numpy as np

if __name__ == '__main__':
    model = fasttext.load_model("data/en-de/cc.de.300.bin")
    word_count_file = "data/de.sorted.word_count"
    word_limit = 25000
    feats = []
    with open(word_count_file) as fi:
        for line in tqdm.tqdm(fi):
            count, word = line.strip().split("\t")
            count = int(count.strip())
            word = word.strip().lower()
            feats.append(model[word])

    np.save('de.fasttext.npy', np.asarray(feats))
