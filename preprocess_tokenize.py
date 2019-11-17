# import fasttext
import tqdm
from polyglot.text import Text

if __name__ == '__main__':
    # model = fasttext.load_model("data/cc.de.300.bin")
    tsv_file = "data/en-de/features/train/feats/feat.tsv"
    output_tsv_file = "data/en-de/features/train/feats/feat.tokenized.tsv"
    with open(tsv_file) as fi:
        with open(output_tsv_file, 'w') as fo:
            for line in tqdm.tqdm(fi):
                l1, l2, feat_file, wav_file, start, duration = line.strip().split("\t")
                sentence_feats = []
                l1_text = Text(l1, hint_language_code="en")
                l2_text = Text(l2, hint_language_code="de")
                fo.write('\t'.join([l1, l2, feat_file, wav_file, start, duration, ' '.join(l1_text.words), ' '.join(l2_text.words)]) + "\n")
