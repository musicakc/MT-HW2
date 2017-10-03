'''
IBM Model 1 implementation for Machine Translation class HW2 Aligner
'''

#!/usr/bin/env python
import math
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with IBM Model 1...")

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
'''
fprobs = defaultdict(float) #prob(f)
eprobs = defaultdict(float) #prob(e)
thetak_prob = defaultdict(float) #Theta prob (t)
countfe = defaultdict(float) #Counter (fe_count)
counte = defaultdict(float) #coutner (total_e)
countf = defaultdict(float)
sumtotal = defaultdict(float)#total_prob for a fword
n = 0
# add [:opts.num_sents] to end of for to limit training data to input
#crating the vocabularies
fvocab = set()
evocab = set()
for number in range(opts.num_sents):
        sentence = [sentence for sentence in bitext[number]]
        fvocab.update(set(sentence[0]))
        evocab.update(set(sentence[1]))
# Assigning initial uniform probabilities
for fword in fvocab:
    for eword in evocab:
            thetak_prob[fword,eword] = 1 / float(len(evocab))
sys.stderr.write("Training...\n")

# Actual training begins here
for _ in range (1):
  for (fsentence, esentence) in bitext:
    if n % 500 == 0:
      sys.stderr.write("%i sentences read\n" %n)
    n = n + 1
    for fword in fsentence:
      if fword not in fprobs:
        fprobs[fword] = {}
        sumtotal[fword] = 0.0
      total_sum = 0.0
      for eword in esentence: 
        if eword not in eprobs:
          eprobs[eword] = 1/float(len(esentence))
        else:
          eprobs[eword] += 1/float(len(esentence))
        if eword not in (fprobs[fword]):
          fprobs[fword][eword] = eprobs[eword]
        else:
          fprobs[fword][eword] += eprobs[eword]
        total_sum += fprobs[fword][eword]
        sumtotal[fword] = total_sum
      for eword in esentence:
        c = fprobs[fword][eword]/ float((total_sum))
        if (fword,eword) in countfe:
          countfe[(fword,eword)] += c
        else:
          countfe[(fword,eword)] = c
        if eword in counte:
          counte[eword] += c
        else:
          counte[eword] = c  
    for (fword, eword) in countfe:
        thetak_prob[(fword, eword)] += countfe[(fword, eword)]/ counte[eword]
        #normalizing thetak_prob
        thetak_prob[(fword,eword)] /= sumtotal[fword]

# Predict/Align: Most Probable Alignment:
# Alignment Step
sys.stderr.write("\nAligning...\n")
n = 0
for (fsentence, esentence) in bitext:
  if n % 500 == 0:
    sys.stderr.write("%s? sentences aligned\n" %n)
  fnum = -1
  for fword in fsentence:
      fnum = fnum + 1
      best_j = ''
      best_prob = 0.0
      best_num = []
      enum = -1
      for eword in esentence:
       enum  = enum + 1 
       if eword is not None:
        if thetak_prob[(fword, eword)] > best_prob:
          #if aligns to null, dont print
            best_prob = thetak_prob[(fword, eword)]
            best_j = eword
            best_num = [enum]
       elif thetak_prob[(fword, eword)] == best_prob:
          #if aligns to null, dont print
            best_num.append(enum)
      for num in range(len(best_num)):
         if math.fabs(best_num[num] - fnum) < 4:
            sys.stdout.write("%i-%i " % (fnum,best_num[num]))
  sys.stdout.write("\n")
  n +=1
'''

#Initialise n which is used to store number of sentences
n = 0

#Count of french words
f_count = defaultdict(float)
#Count of english words
e_count = defaultdict(float)

fe_count = defaultdict(float)
sum_t = defaultdict(float)
total = defaultdict(float)

#Theta probability
t_prob = defaultdict(float)
#Probability of french words
f_prob = defaultdict(float)
#Probability of english words
e_prob = defaultdict(float)

#Update vocabularies from bitext
f_vocab = set()
e_vocab = set()
for i in range(opts.num_sents):
  sen = [sen for sen in bitext[i]]
  f_vocab.update(set(sen[0]))
  e_vocab.update(set(sen[1]))

#Initialise theta probablity values
for f in f_vocab:
  for e in e_vocab:
    t_prob[f,e] = 1 / float(len(e_vocab))


#Calculating probabilities according to the sentences in vocabulary
#stored in bitext.

for _ in range(1):
  for (f_sen, e_sen) in bitext:
    if n % 500 == 0:
      sys.stderr.write("%i sentences have been read \n" %n)
    n = n + 1
    for f in f_sen:
      if f not in f_prob:
        f_prob[f] = {}
        sum_t[f] = 0.0
      total = 0.0
      for e in e_sen:
        if e not in e_prob:
          e_prob[e] = (1 / float(len(e_sen)))
        else:
          e_prob[e] += (1 / float(len(e_sen)))
        if e not in (f_prob[f]):
          f_prob[f][e] = e_prob[e]
        else:
          f_prob[f][e] += e_prob[e]
        total += f_prob[f][e]
        sum_t[f] = total
      for e in e_sen:
        count = f_prob[f][e] / float(total)
        if (f,e) in fe_count:
          fe_count[(f,e)] += count 
        else:
          fe_count[(f,e)] = count
        if e in e_count:
          e_count[e] += count
        else:
          e_count[e] = count


    #Calculating actual theta probabilities
    for (f, e) in fe_count:
      t_prob[(f,e)] += fe_count[(f,e)] / e_count[e]
      t_prob[(f,e)] /= sum_t[f] 

sys.stderr.write("\n Aligning... \n")
n = 0
for (f_sen, e_sen) in bitext:
  if n % 500 == 0:
    sys.stderr.write("%s? sentences aligned\n" %n)
  fnum = -1
  for f in f_sen:
    fnum = fnum + 1
    best_a = ''
    best_p = 0.0
    best_num = []
    enum = -1
    for e in e_sen:
      enum = enum + 1
      if e is not None:
        if t_prob[(f,e)] > best_p:
          best_p = t_prob[(f,e)]
          best_a = e
          best_num = [enum]
        elif t_prob[(f,e)] == best_p:
          best_num.append(enum)
      for m in range(len(best_num)):
        if math.fabs(best_num[m] - fnum) < 5:
           sys.stdout.write("%i-%i " % (fnum,best_num[m]))
  sys.stdout.write("\n")
  n += 1