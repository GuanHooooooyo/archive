from transformers import BertTokenizer
from IPython.display import clear_output
import random

PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
clear_output()
vocab = tokenizer.vocab
print("字典大小：", len(vocab))
random_token = random.sample(list(vocab),10)
random_ids = [vocab[t] for t in random_token]
print("{0:20}{1:15}".format("token", "index"))
print("-" * 25)

for t,id in zip(random_token,random_ids):
    print("{0:15}{1:10}".format(t, id))
print('----------')

indices = list(range(647, 657))
pairs = [(t,id) for t,id in vocab.items() if id in indices]
for pair in pairs:
    print(pair)

##test tokens in chinese word
text = "人是最[MASK]的風景"
token = tokenizer.tokenize(text)
ids = tokenizer.convert_ids_to_tokens(token)
print(text)
print(token[:7])
print(ids[:7])
