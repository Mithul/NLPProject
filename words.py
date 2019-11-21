import pickle

import os, tqdm

data_directory = './data'
data_directory = os.path.join(data_directory,"en-de")
feats = os.path.join(data_directory, "features/train/feats/feat.tokenized.tsv")	
l1_dic = {}
l2_dic = {}

def add_to(dic,val):
	if val not in dic:
		dic[val] = 1
	else:
		dic[val]+=1

for i, line in enumerate(open(feats)):
	_, _, _, _, _, _, l1_s, l2_s = line.split("\t")
	for word in l2_s.strip().lower().split(" "):
		add_to(l2_dic,word)
	for word in l1_s.strip().lower().split(" "):
		add_to(l1_dic,word)

l1_common = set()
l2_common = set()
threshold = 10000

sorted_l1 = sorted(l1_dic.items(), key=lambda kv: kv[1])
sorted_l2 = sorted(l2_dic.items(), key=lambda kv: kv[1])

l1_common = set([i[0] for i in sorted_l1[-threshold:]])
l2_common = set([i[0] for i in sorted_l2[-threshold:]])


with open('l1_common', 'wb') as fp:
    pickle.dump(l1_common, fp)


with open('l2_common', 'wb') as fp:
    pickle.dump(l2_common, fp)
