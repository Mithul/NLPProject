import tqdm, os, sys
from polyglot.text import Text
import fasttext
import numpy as np

def preprocess_tokenize(dataset_type):
    tsv_file = "data/en-de/features/" + dataset_type + "/feats/feat.tsv"
    output_tsv_file = "data/en-de/features/" + dataset_type + "/feats/feat.tokenized.tsv"
    with open(tsv_file) as fi:
        with open(output_tsv_file, 'w') as fo:
            for line in tqdm.tqdm(fi):
                l1, l2, feat_file, wav_file, start, duration = line.strip().split("\t")
                sentence_feats = []
                l1_text = Text(l1, hint_language_code="en")
                l2_text = Text(l2, hint_language_code="de")
                fo.write('\t'.join([l1, l2, feat_file, wav_file, start, duration, ' '.join(l1_text.words), ' '.join(l2_text.words)]) + "\n")


def preprocess_fasttext_matrix():
    model = fasttext.load_model("data/en-de/cc.de.300.bin")
    word_count_file = "data/en-de/de.words"
    feats = []
    with open(word_count_file) as fi:
        for line in tqdm.tqdm(fi):
            word = line.strip()
            # count = int(count.strip())
            word = word.strip().lower()
            feats.append(model[word])

    np.save('de.fasttext.npy', np.asarray(feats))


if __name__ == '__main__':
    # preprocess_tokenize("train")
    # data_dir = '/'.join(sys.argv[0].split("/")[:-1])
    # cmd = "cd "+ data_dir + " && bash get_top_words.sh"
    # print("Executing command ", cmd)
    # os.system(cmd)
    preprocess_fasttext_matrix()
