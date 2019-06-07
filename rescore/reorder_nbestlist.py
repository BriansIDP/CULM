'''This script reorders the nbestlist for dev and eval set in time order'''

import sys, os
from operator import itemgetter

meeting_dict = {}

def sort_and_write(elems):
    if len(elems) == 0:
        return []
    elems.sort(key=itemgetter(0))
    sub_to_write = []
    for elem in elems:
        sub_to_write.append(elem[1])
    return sub_to_write

listfile = sys.argv[1]
with open(listfile) as fin:
    elem_to_sort = []
    to_write = []
    for line in fin:
        line_elems = line.strip().split('/')
        meeting = line_elems[-2]
        start = int(line_elems[-1][-15:-8])
        if meeting not in meeting_dict:
            meeting_dict[meeting] = [(start, line)]
        else:
            meeting_dict[meeting].append((start, line))

for meeting, elem in meeting_dict.items():
    to_write += sort_and_write(elem)

with open('time_sorted_' + listfile, 'w') as fout:
    fout.writelines(to_write)
