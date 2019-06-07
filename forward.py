# coding: utf-8
import argparse
import sys, os
import torch
import math
from operator import itemgetter

import data

parser = argparse.ArgumentParser(description='PyTorch Level-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='model.pt',
                    help='location of the 1st level model')
parser.add_argument('--model2', type=str, default='model.pt',
                    help='location of the interpolate model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--reset', action='store_true',
                    help='reset at sentence boundaries')
parser.add_argument('--memorycell', action='store_true',
                    help='Use memory cell as input, otherwise use output cell')
parser.add_argument('--uttlookback', type=int, default=1,
                    help='Number of backward utterance embeddings to be incorporated')
parser.add_argument('--uttlookforward', type=int, default=1,
                    help='Number of forward utterance embeddings to be incorporated')
parser.add_argument('--excludeself', type=int, default=1,
                    help='current utterance embeddings to be incorporated')
parser.add_argument('--saveprefix', type=str, default='tensors/AMI',
                    help='Specify which data utterance embeddings saved')
parser.add_argument('--nbest', type=str, default='dev.nbest.info.txt',
                    help='Specify which nbest file to be used')
parser.add_argument('--function', type=str, default='embedding',
                    help='Specify which function to be used: embeddings or nbest')
parser.add_argument('--rnnscale', type=float, default=6,
                    help='how much importance to attach to rnn score')
parser.add_argument('--lm', type=str, default='original',
                    help='Specify which language model to be used: rnn, ngram or original')
parser.add_argument('--ngram', type=str, default='dev_ngram.st',
                    help='Specify which ngram stream file to be used')
parser.add_argument('--saveemb', action='store_true',
                    help='save utterance embeddings')
parser.add_argument('--save1best', action='store_true',
                    help='save 1best list')
parser.add_argument('--context', type=str, default='0',
                    help='Specify which utterance embeddings to be used')
parser.add_argument('--logfile', type=str, default='LOGs/log.txt',
                    help='the logfile for this script')
parser.add_argument('--interp', action='store_true',
                    help='Linear interpolation of LMs')
parser.add_argument('--factor', type=float, default=0.8,
                    help='ngram interpolation weight factor')
parser.add_argument('--gscale', type=float, default=12.0,
                    help='ngram grammar scaling factor')
args = parser.parse_args()

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# Read in dictionary
logging("Reading dictionary...")
dictionary = {}
with open(os.path.join(args.data, 'dictionary.txt')) as vocabin:
    lines = vocabin.readlines()
    for line in lines:
        ind, word = line.strip().split(' ')
        if word not in dictionary:
            dictionary[word] = ind
        else:
            logging("Error! Repeated words in the dictionary!")

ntokens = len(dictionary)
eosidx = int(dictionary['<eos>'])

device = torch.device("cuda" if args.cuda else "cpu")

# Read in trained 1st level model
logging("Reading model...")
with open(args.model, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
    if args.cuda:
        model.cuda()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_utt_embeddings(model): 
    """ This is the function to write out the utterance embeddings."""
    model.eval()
    model.set_mode('eval')
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for setname in ['train', 'valid', 'test']: 
            with open(os.path.join(args.data, setname+'.txt')) as fin:
                lines = fin.readlines()
                Nlines = len(lines)
                utt_embeddings = []
                totalfile = []
                filecontext = []
                for ind, line in enumerate(lines):
                    currentline = []
                    context = []
                    for word in line.strip().split(' '):
                        if word in dictionary:
                            currentline.append(int(dictionary[word]))
                        else:
                            currentline.append(int(dictionary['OOV']))
                    currentline.append(eosidx)

                    # Forward starts here
                    input = torch.LongTensor([eosidx]+currentline)
                    input = input.view(1, -1).t().to(device)
                    rnnout, hidden = model(input, hidden, outputflag=1)
                    totalfile += currentline
                    filecontext += [ind for i in range(len(currentline))]
                    if args.memorycell:
                        utt_embeddings.append(hidden[1])
                    else:
                        utt_embeddings.append(rnnout)
                    hidden = model.init_hidden(1)
                    # repackage_hidden(hidden)
                    if ind % 1000 == 0 and ind != 0:
                        logging('{}/{} completed'.format(ind, len(lines)))
            totalfile = [eosidx] + totalfile
            filecontext.append(filecontext[-1])
            torch.save(torch.cat(utt_embeddings, 0), args.saveprefix+setname+'_utt_embed.pt')
            torch.save(torch.LongTensor(totalfile), args.saveprefix+setname+'_fullind.pt')
            torch.save(torch.LongTensor(filecontext), args.saveprefix+setname+'_embind.pt')

# Forward each sentence in the nbest list
def forward_each_utterance(model, line, forwardCrit, utt_idx, ngram_logprobs, aux_in, prev_emb):
    """ After reading in the nbestlist file for one utterance
        this function chooses the best according to certain lms. """
    # Use zero initial hidden states
    hidden = model.init_hidden(1)
    # Use final hidden state of the previous best utterance
    # hidden = prev_emb

    # Parse the nbest list file
    linevec = line.strip().split()
    acoustic_score = float(linevec[0])
    lmscore = float(linevec[1])
    emb = None
    utterance = linevec[4:-1]

    # Convert the current line to index representations
    currentline = []
    for i, word in enumerate(utterance):
        if word in dictionary:
            currentline.append(int(dictionary[word]))
        else:
            currentline.append(int(dictionary['OOV']))

    # Forward each utterance
    targets = torch.LongTensor(currentline + [eosidx]).to(device)
    input = torch.LongTensor([eosidx] + currentline).to(device)

    # Forward each utterance for rnn
    if args.lm == 'rnn':
        input = input.view(1, -1).t()
        output, hidden = model(input, hidden)
        if args.interp:
            rnnlogProbs = forwardCrit(output.view(-1, ntokens), targets)
            rnnProbs = torch.exp(-rnnlogProbs)
            ngram_logprobs = torch.tensor([float(prob)/args.gscale for prob in ngram_logprobs])
            ngramProbs = torch.exp(ngram_logprobs).to(device)
            rnnProbs = rnnProbs * args.factor + ngramProbs * (1 - args.factor)
            rnnProbs = torch.log(rnnProbs)
            rnnscore = - float(rnnProbs.sum())
        else:
            logProb = forwardCrit(output.view(-1, ntokens), targets)
            rnnscore = float(logProb * len(currentline))
        # Calculate total score
        total_score = - rnnscore * args.rnnscale + acoustic_score
        out = '\t'.join([str(utt_idx), str(acoustic_score), '{:5.2f}'.format(rnnscore), '{:5.2f}'.format(total_score), ' '.join(utterance)+' <eos>\n'])
        emb = hidden

    # Forward each utterance for transformer lm
    if args.lm == 'transformer':
        # TO be filled in
        raise

    return out, total_score, utterance, emb

    # elif args.lm == 'curnn':
    #     # Forward nbest list for cross-utterance rnn
    #     input = input.view(1, -1).t()
    #     n = input.size(0)
    #     # Expand the auxiliary input feature
    #     aux_in = aux_in.repeat(n, 1).view(n, 1, -1)
    #     output, hidden, penalty = model(input, aux_in, hidden, eosidx=eosidx, device=device)
    #     logProb = forwardCrit(output.view(-1, ntokens), targets)
    #     if args.interp:
    #         rnnlogProbs = forwardCrit(output.view(-1, ntokens), targets)
    #         ngram_logprobs = torch.tensor([float(prob)/args.gscale for prob in ngram_logprobs])
    #         rnnProbs = torch.exp(-rnnlogProbs) * args.factor + torch.exp(-ngram_logprobs) * (1 - args.factor)
    #         rnnscore = - float(torch.log(rnnProbs).sum())
    #     else:
    #         rnnscore = float(logProb * len(currentline))
    #     # Calculate total score
    #     total_score = - rnnscore * args.rnnscale + acoustic_score
    #     out = '\t'.join([str(utt_idx), str(acoustic_score), '{:5.2f}'.format(rnnscore), '{:5.2f}'.format(total_score), ' '.join(utterance)+' <eos>\n'])
        # import pdb; pdb.set_trace()
        
    # elif args.lm == 'ngram':
    #      ngram_utt_prob = ngram_probs[ngram_cursor:ngram_cursor+ngram_count]
    #      ngram_cursor += ngram_count
    #      OOV_number = len(utterance) - ngram_count + 1
    #      ngram_sent_prob = -20.0 * OOV_number
    #      for wordprob in ngram_utt_prob:
    #          ngram_sent_prob += math.log(float(wordprob.strip()))
    #      total_score = ngram_sent_prob * args.rnnscale + acoustic_score
    #      out = '\t'.join([str(utt_idx), str(acoustic_score), '{:5.2f}'.format(ngram_sent_prob), '{:5.2f}'.format(total_score), ' '.join(utterance)+' <eos>\n'])

def forward_nbest_utterance(model, nbestfile):
    """ The main body of the rescore function. """
    model.eval()
    model.set_mode('eval')
    # initialise the one best embedding
    best_emb = model.init_hidden(1)
    # decide if we calculate the average of the loss
    if args.interp:
        forwardCrit = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        forwardCrit = torch.nn.CrossEntropyLoss()

    # initialising variables needed
    ngram_cursor = 0
    lmscored_lines = []
    best_utt_list = []
    emb_list = []
    utt_idx = 0
    ngram_probs = []

    # Ngram used for lattice rescoring
    if args.interp:
        ngram_listfile = open(args.ngram)
    if args.lm == 'curnn':
        embeddings = torch.load(nbestfile+'_utt_embed.pt')
        context_shift = [int(i) for i in args.context.strip().split(' ')]
    with open(nbestfile) as filein:
        with torch.no_grad():
            for utterancefile in filein:
                labname = utterancefile.strip().split('/')[-1]
                labname = labname + '.rec'
                # Fill in contexts for utterance embeddings indexing
                current_aux_in = None
                # if args.lm == 'curnn':
                #     current_aux_in = []
                #     for i in context_shift:
                #         if i + utt_idx < 0:
                #             current_aux_in.append(embeddings[0])
                #         elif i + utt_idx >= embeddings.size(0):
                #             current_aux_in.append(embeddings[-1])
                #         else:
                #             current_aux_in.append(embeddings[i+utt_idx])
                #     current_aux_in = torch.cat(current_aux_in, 1)

                # Read in ngram LM files for interpolation
                if args.interp:
                    ngram_probfile_name = ngram_listfile.readline()
                    ngram_probfile = open(ngram_probfile_name.strip())
                    ngram_prob_lines = ngram_probfile.readlines()

                # Start processing each nbestlist
                with open(utterancefile.strip()) as uttfile:
                    uttscore = []
                    for i, line in enumerate(uttfile):
                        if args.interp:
                            ngram_elems = ngram_prob_lines[i].strip().split(' ')
                            sent_len = int(ngram_elems[0])
                            ngram_probs = ngram_elems[sent_len+2:]

                        outputline, score, utt, emb = forward_each_utterance(model, line, forwardCrit, utt_idx, ngram_probs, current_aux_in, best_emb)
                        lmscored_lines.append(outputline)
                        uttscore.append((utt, score, emb))

                    utt_idx += 1
                    # Get the best utterance by sorting
                    bestutt_group = max(uttscore, key=itemgetter(1))
                    bestutt = bestutt_group[0]
                    best_emb = bestutt_group[2]
                    best_utt_list.append((labname, bestutt))
                    emb_list.append(best_emb[1])
                # Log every completion of 500 utterances
                if utt_idx % 500 == 0:
                    logging(str(utt_idx))
    # Write out renewed lmscore file
    with open(nbestfile+'.renew.'+args.lm, 'w') as fout:
        fout.writelines(lmscored_lines)
    # Save 1-best for later use for the context
    if args.save1best:
        # Write out for second level forwarding
        with open(nbestfile+'.context', 'w') as fout:
            for i, eachutt in enumerate(best_utt_list):
                linetowrite = '<eos> ' + ' '.join(eachutt[1]) + ' <eos>\n'
                fout.write(linetowrite)
    # Save 1-best file for scoring, in .rec file format
    with open(nbestfile + '.1best.'+args.lm, 'w') as fout:
        fout.write('#!MLF!#\n')
        for eachutt in best_utt_list:
            labname = eachutt[0]
            start = 100000
            end = 200000
            fout.write('\"'+labname+'\"\n')
            for eachword in eachutt[1]:
                if eachword[0] == '\'':
                    eachword = '\\' + eachword
                fout.write(str(start) + ' ' + str(end) + ' ' + eachword+'\n')
                start += 100000
                end += 100000
            fout.write('.\n')
    # Save utterance embeddings if necessary
    if args.saveemb:
        torch.save(torch.cat(emb_list, 0), nbestfile+'_utt_embed.pt')


def writeout_logp(model):
    """ This function write out the log probabilities 
        for each words in test set."""
    model.eval()
    model.set_mode('eval')
    forwardCrit = torch.nn.CrossEntropyLoss(reduction='none')
    hidden = model.init_hidden(1)
    lines_to_write = []
    idx = 0
    totalloss = 0.
    total_symbol = 0
    if args.lm == 'curnn':
        embeddings = torch.load(args.saveprefix+'valid_utt_embed.pt')
        context_shift = [int(i) for i in args.context.strip().split(' ')]
    with open(os.path.join(args.data, 'valid.txt')) as fin:
        with torch.no_grad():
            # hidden = model.init_hidden(1)
            for line in fin:
                hidden = model.init_hidden(1)
                utterance = line.strip().split()
                currentline = []
                for i, word in enumerate(utterance):
                    if word in dictionary:
                        currentline.append(int(dictionary[word]))
                    else:
                        currentline.append(int(dictionary['OOV']))
                currentline = [eosidx] + currentline
                currenttarget = currentline[1:]
                currenttarget.append(eosidx)
                targets = torch.LongTensor(currenttarget).to(device)
                input = torch.LongTensor(currentline).to(device)
                input = input.view(1, -1).t()
                if args.lm == 'rnn':
                    output, hidden = model(input, hidden)
                    logProbs = forwardCrit(output.view(-1, ntokens), targets)
                elif args.lm == 'curnn':
                    current_aux_in = []
                    for i in context_shift:
                        if i + idx < 0:
                            current_aux_in.append(embeddings[0])
                        elif i + idx >= embeddings.size(0):
                            current_aux_in.append(embeddings[-1])
                        else:
                            current_aux_in.append(embeddings[i+idx])
                    current_aux_in = torch.cat(current_aux_in, 1)
                    n = input.size(0)
                    # Expand the auxiliary input feature
                    aux_in = current_aux_in.repeat(n, 1).view(n, 1, -1)
                    output, hidden, penalty = model(input, aux_in, hidden, eosidx=eosidx, device=device)
                    # hidden2 = model.init_hidden(1)
                    # output2, hidden2 = model2(input, hidden2)
                    logProbs = forwardCrit(output.view(-1, ntokens), targets)
                    # logProbs2 = forwardCrit(output2.view(-1, ntokens), targets)
                    # rnnProbs = torch.exp(-logProbs1) * args.factor + torch.exp(-logProbs2) * (1 - args.factor)
                    # logProbs = - torch.log(rnnProbs)
                prob_list = []
                totalloss += logProbs.sum()
                total_symbol += len(logProbs)
                for probs in logProbs.tolist():
                    prob_list.append('{:2.2f}'.format(probs))
                utterancestr = ' '.join(utterance + ['</s>']) + '\n'
                logProbstr = ' '.join(prob_list) + '\n'
                lines_to_write.append(utterancestr)
                lines_to_write.append(logProbstr)
                if idx != 0 and idx % 1000 == 0:
                    logging(idx)
                idx += 1
    with open(args.lm+'logprob_dev.txt', 'w') as fout:
        fout.writelines(lines_to_write)
    logging(torch.exp(totalloss/total_symbol))

logging('getting utterances')
if args.function == 'embedding':
# get_utt_embedding_groups(model)
    get_utt_embeddings(model)
elif args.function == 'nbest':
    forward_nbest_utterance(model, args.nbest)
elif args.function == 'writeout':
    writeout_logp(model)
