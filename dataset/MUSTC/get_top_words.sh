cut -f 8 en-de/features/train/feats/feat.tokenized.tsv | sed "s: :\n:g" | tr "[:upper:]" "[:lower:]" | sort | uniq -c | sort -nr | sed -r "s:^\s+::g" | sed "s: :\t:g" > en-de/de.sorted.word_count
head -n 25000 en-de/de.sorted.word_count | cut -f 2 > en-de/de.words
