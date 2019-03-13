import sys, os
from torch.utils.data import Dataset, DataLoader
import torch

class Dictionary(object):
    def __init__(self, dictfile):
        self.word2idx = {}
        self.idx2word = []
        self.unigram = []
        self.letter_trigram, self.count = self.build_letter_trigram()
        self.build_dict(dictfile)

    def build_letter_trigram(self):
        letter_trigram = {}
        upperalpha_extended = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ!\'-'
        upperalpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ\'-.'
        upperalpha_ending = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ$\'.'
        count = 0
        for i in upperalpha_extended:
            if i not in letter_trigram:
                letter_trigram[i] = {}
            for j in upperalpha:
                if j not in letter_trigram[i]:
                    letter_trigram[i][j] = {}
                for k in upperalpha:
                    letter_trigram[i][j][k] = count
                    count += 1
        return letter_trigram, count

    def get_trigram(self, idx):
        word = self.idx2word[idx]
        extendedword = '!' + word + '$'
        letter_vec = torch.zeros(self.count)
        if idx != self.get_eos() and idx != self.get_sos():
            for i in range(len(extendedword)-3):
                index = self.letter_trigram[extendedword[i]][extendedword[i+1]][extendedword[i+2]]
                letter_vec[index] = 1
        return torch.tensor(letter_vec).view(1, -1)

    def build_dict(self, dictfile):
        with open(dictfile, 'r', encoding="utf8") as f:
            for line in f:
                index, word = line.strip().split(' ')
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.unigram.append(0)
        self.unigram[self.word2idx[word]] += 1
        return self.word2idx[word]

    def get_eos(self):
        return self.word2idx['<eos>']

    def get_sos(self):
        return self.word2idx['<sos>']

    def normalize_counts(self):
        self.unigram /= np.sum(self.unigram)
        self.unigram = self.unigram.tolist()

    def __len__(self):
        return len(self.idx2word)

class LMdata(Dataset):
    def __init__(self, data_file, dictionary, individual_utt = False):
        '''Load data_file'''
        self.data_file = data_file
        self.data = []
        with open(self.data_file, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                self.data += words
        self.dictionary = dictionary
        self.individual_utt = individual_utt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data[idx] in self.dictionary.word2idx:
            return self.dictionary.word2idx[self.data[idx]]
        else:
            return self.dictionary.word2idx['OOV']

def collate_fn(batch):
    return torch.LongTensor(batch)

def create(datapath, dictfile, batchSize=1, shuffle=False, workers=0):
    loaders = []
    dictionary = Dictionary(dictfile)
    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(datapath, '%s.txt' %split)
        dataset = LMdata(data_file, dictionary)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
    return loaders[0], loaders[1], loaders[2], dictionary

if __name__ == "__main__":
    datapath = sys.argv[1]
    dictfile = sys.argv[2]
    traindata, valdata, testdata = create(datapath, dictfile, batchSize=1000000, workers=0)
    for i_batch, sample_batched in enumerate(traindata):
        print(i_batch, sample_batched.size())
