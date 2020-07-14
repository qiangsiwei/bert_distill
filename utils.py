# -*- coding: utf-8 -*-

import jieba, random, fileinput, numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


def load_data(name):
    def get_w2v():
        for line in open('data/cache/word2vec', encoding="utf-8").read().strip().split('\n'):
            line = line.strip().split()
            if not line: continue
            yield line[0], np.array(list(map(float, line[1:])))

    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
    texts = [' '.join(jieba.cut(line.split('\t', 1)[1].strip())) \
             for line in open('data/{}/{}.txt'.format(name, name), encoding="utf-8",
                              ).read().strip().split('\n')]
    tokenizer.fit_on_texts(texts)
    # with open('word2vec','w') as out:
    # 	for line in fileinput.input('sgns.sogou.word'):
    # 		word = line.strip().split()[0]
    # 		if word in tokenizer.word_index:
    # 			out.write(line+'\n')
    # 	fileinput.close()
    x_train, y_train = [], [];
    text_train = []
    for line in open('data/{}/train.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_train.append(text.strip())
        x_train.append(' '.join(jieba.cut(text.strip())))
        y_train.append(int(label))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev, y_dev = [], []
    text_dev = []
    for line in open('data/{}/dev.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_dev.append(text.strip())
        x_dev.append(' '.join(jieba.cut(text.strip())))
        y_dev.append(int(label))
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test, y_test = [], []
    text_test = []
    for line in open('data/{}/test.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_test.append(text.strip())
        x_test.append(' '.join(jieba.cut(text.strip())))
        y_test.append(int(label))
    x_test = tokenizer.texts_to_sequences(x_test)
    v_size = len(tokenizer.word_index) + 1
    embs, w2v = np.zeros((v_size, 300)), dict(get_w2v())
    for word, index in tokenizer.word_index.items():
        if word in w2v: embs[index] = w2v[word]
    return (x_train, y_train, text_train), \
           (x_dev, y_dev, text_dev), \
           (x_test, y_test, text_test), \
           v_size, embs

def load_data_aug(name, n_iter=20, p_mask=0.1, p_ng=0.25, ngram_range=(3,6)):
    def get_w2v():
        for line in open('data/cache/word2vec', encoding="utf-8").read().strip().split('\n'):
            line = line.strip().split()
            if not line: continue
            yield line[0], np.array(list(map(float, line[1:])))

    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
    texts = [' '.join(jieba.cut(line.split('\t', 1)[1].strip())) \
             for line in open('data/{}/{}.txt'.format(name, name), encoding="utf-8",
                              ).read().strip().split('\n')]
    tokenizer.fit_on_texts(texts)

    x_train, y_train = [], [];
    text_train = []
    for line in open('data/{}/train.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text = text.strip()
        # preserve one original sample first
        text_train.append(text)
        x_train.append(' '.join(jieba.cut(text)))
        y_train.append(int(label))
        # data augmentation
        used_texts = {text}
        for i in range(n_iter):
            words = jieba.lcut(text)
            # word masking
            words = [x if np.random.rand() < p_mask else "[MASK]" for x in words]
            # n-gram sampling
            if np.random.rand() < p_ng:
                n_gram_len = np.random.randint(ngram_range[0], ngram_range[1]+1)
                n_gram_len = min(n_gram_len, len(words))
                n_gram_start = np.random.randint(0, len(words)-n_gram_len+1)
                words = words[n_gram_start:n_gram_start+n_gram_len]
            new_text = "".join(words)
            if new_text not in used_texts:
                text_train.append(new_text)
                x_train.append(' '.join(words))
                y_train.append(int(label))
                used_texts.add(new_text)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev, y_dev = [], []
    text_dev = []
    for line in open('data/{}/dev.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_dev.append(text.strip())
        x_dev.append(' '.join(jieba.cut(text.strip())))
        y_dev.append(int(label))
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test, y_test = [], []
    text_test = []
    for line in open('data/{}/test.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_test.append(text.strip())
        x_test.append(' '.join(jieba.cut(text.strip())))
        y_test.append(int(label))
    x_test = tokenizer.texts_to_sequences(x_test)
    v_size = len(tokenizer.word_index) + 1
    embs, w2v = np.zeros((v_size, 300)), dict(get_w2v())
    for word, index in tokenizer.word_index.items():
        if word in w2v: embs[index] = w2v[word]
    return (x_train, y_train, text_train), \
           (x_dev, y_dev, text_dev), \
           (x_test, y_test, text_test), \
           v_size, embs

if __name__ == '__main__':
    # load_data(name='hotel')
    load_data_aug(name='hotel')
