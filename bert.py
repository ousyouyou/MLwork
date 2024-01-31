import random
import glob
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel

# BERTの日本語モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
from transformers import BertTokenizer, BertForPreTraining

# 10-5
# カテゴリーのリスト
category_list = [
    'technologie',
    'sport',
    'space',
    'politics',
    'medical',
    'historical',
    'graphics',
    'food',
    'entertainment',
    'business'
]

# トークナイザとモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForPreTraining.from_pretrained('bert-base-cased')
model = model.cuda()

# 各データの形式を整える
max_length = 256
sentence_vectors = [] # 文章ベクトルを追加していく。
labels = [] # ラベルを追加していく。
for label, category in enumerate(tqdm(category_list)):
    for file in glob.glob(f'./{category}/{category}*'):
        # 記事から文章を抜き出し、符号化を行う。
        lines = open(file).read().splitlines()
        text = '\n'.join(lines[3:])
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoding = { k: v.cuda() for k, v in encoding.items() }
        attention_mask = encoding['attention_mask']

        # 文章ベクトルを計算
        # BERTの最終層の出力を平均を計算する。（ただし、[PAD]は除く。）
        with torch.no_grad():
            output = model(**encoding)
            last_hidden_state = output[0]
            averaged_hidden_state = \
                (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
                / attention_mask.sum(1, keepdim=True)

        # 文章ベクトルとラベルを追加
        sentence_vectors.append(averaged_hidden_state[0].cpu().numpy())
        labels.append(label)

# それぞれをnumpy.ndarrayにする。
sentence_vectors = np.vstack(sentence_vectors)
labels = np.array(labels)


plt.figure(figsize=(10,10))
for label in range(9):
    plt.subplot(3,3,label+1)
    index = labels == label
    plt.plot(
        sentence_vectors_pca[:,0],
        sentence_vectors_pca[:,1],
        'o',
        markersize=1,
        color=[0.7, 0.7, 0.7]
    )
    plt.plot(
        sentence_vectors_pca[index,0],
        sentence_vectors_pca[index,1],
        'o',
        markersize=2,
        color='k'
    )
    plt.title(category_list[label])