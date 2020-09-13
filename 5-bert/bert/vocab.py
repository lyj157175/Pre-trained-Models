from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import tqdm
from collections import Counter

from torch.utils.data import Dataset
import tqdm
import torch
import random
import torch.nn.functional as F



# 配置
class Args(object):
    def __init__(self):
        self.train_dataset='../data/corpus.small'
        self.test_dataset='../data/corpus.small'
        self.vocab_path='../data/corpus.small.vocab'
        self.output_path='../data/corpus.small.vocab'
        self.hidden=256
        self.layers=8
        self.attn_heads=8
        self.seq_len=20
        self.batch_size=64
        self.epochs=10
        self.num_workers=5
        self.with_cuda=True
        self.log_freq=10
        self.corpus_lines=""
        self.lr=1e-3
        self.adam_weight_decay=0.01
        self.adam_beta1=0.9
        self.adam_beta2=0.999
        
args = Args()



#################### vocab ##################
class TorchVocab(object):

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1




class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)




class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab': 
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


############ build vocab ###################
corpus_path=args.train_dataset
with open(corpus_path, 'r') as file:
    vocab = WordVocab(file, max_size=None, min_freq=1)
print("VOCAB SIZE:", len(vocab))
vocab.save_vocab(args.output_path)



############ bert_dataset ##############
class BERTDataset(Dataset):

    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line[:-1].split("\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # t1为句子1
        t1, (t2, is_next_label) = self.datas[item][0], self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)   # 句子1，修改后和真实的句子
        t2_random, t2_label = self.random_word(t2)   # 句子2，修改后和真实的句子

        # [CLS] = SOS, [SEP] = EOS
        # [CLS] 1, 1, 1, 1, 1 [sep] 2, 2, 2, 2, 2 [sep] 
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len] # 超出seq_len的不要
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding) 

        output = {"bert_input": bert_input,        # mask后句子的数字表示
                  "bert_label": bert_label,        # 是否被mask， 否则为0，是则为自己 
                  "segment_label": segment_label,  # 句子1标记为1，句子2标记为2
                  "is_next": is_next_label}        # 是否为下一句 

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        # token需要更改的句子，与output_label的长度保持一致
        tokens = sentence.split()
        output_label = []  # 真实的label

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                # 80% randomly change token to make token
                if prob < prob * 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob * 0.8 <= prob < prob * 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                elif prob >= prob * 0.9:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        if random.random() > 0.5:
            return self.datas[index][1], 1
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0


############## build bert_dataset and bert_dataloader ####################
train_dataset = BERTDataset(args.train_dataset, 
                            vocab, 
                            seq_len=args.seq_len, 
                            corpus_lines=args.corpus_lines)
test_dataset = BERTDataset(args.test_dataset, 
                           vocab,
                           seq_len=args.seq_len) 

train_data_loader = DataLoader(train_dataset, 
                               batch_size=args.batch_size, 
                               num_workers=0)
test_data_loader = DataLoader(test_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=0) 


