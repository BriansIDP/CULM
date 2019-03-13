import sys, os

dictionary = {}
ind = 0
for filein in ['train.txt', 'valid.txt', 'test.txt']:
    with open(filein) as fin:
        for line in fin:
            words = line.strip().split(' ')
            words.append('<eos>')
            for word in words:
                if word not in dictionary:
                    dictionary[word] = ind
                    ind += 1
with open('dictionary.txt', 'w') as fout:
    for key, value in dictionary.items():
        line = str(value) + ' ' + key + '\n'
        fout.write(line)
with open('wlist.txt', 'w') as fout:
    for key, value in dictionary.items():
        fout.write(key + '\n')
