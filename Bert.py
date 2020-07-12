from transformers import BertTokenizer, BertModel
from IPython.display import clear_output
import random
import pandas as pd

PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
clear_output()
vocab = tokenizer.vocab
print("字典大小：", len(vocab))
# random_token = random.sample(list(vocab),10)
# random_ids = [vocab[t] for t in random_token]
# print("{0:20}{1:15}".format("token", "index"))
# print("-" * 25)

# for t,id in zip(random_token,random_ids):
#     print("{0:15}{1:10}".format(t, id))
print('----------')

# indices = list(range(647, 657))
# pairs = [(t,id) for t,id in vocab.items() if id in indices]
# for pair in pairs:
#     print(pair)

##test tokens in chinese word
text = "[CLS] 等到潮水 [MASK] 了 就知道誰沒穿褲子"

tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(text)
print(tokens[:15])
print(ids[:15])
##偵測[MASK]並預測填空部分的文字##
from transformers import  BertForMaskedLM
import torch
tokens_tensor = torch.tensor([ids])  # (1, seq_len)
segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
clear_output()
# 使用 masked LM 估計 [MASK] 位置所代表的實際 token
maskedLM_model.eval()
with torch.no_grad():
    outputs = maskedLM_model(tokens_tensor, segments_tensors)
    predictions = outputs[0]
    # (1, seq_len, num_hidden_units)
del maskedLM_model
# 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
masked_index = 5
k = 3
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
#  顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
print("輸入 tokens ：", tokens[:15], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokens[masked_index] = t
    print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens[:15]), '...')


df_train = pd.read_csv('tbrain_train_final_0610.csv')
empty_title = ((df_train['name'].notnull()))

df_train = df_train[~empty_title]
print(df_train.head(5))