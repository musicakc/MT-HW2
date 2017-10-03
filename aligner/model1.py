'''
IBM Model 1 implementation for Machine Translation class HW2 Aligner
'''

#!/usr/bin/env python
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

#Count of french words
f_count = defaultdict(float)
#Count of english words
e_count = defaultdict(float)

fe_count = defaultdict(float)
sum_t = defaultdict(float)

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

#Initialise n which is used to store number of sentences
n = 0

'''Calculating probabilities according to the sentences in vocabulary
stored in bitext.
'''
for _ in range(1):
  for (f_sen, e_sen) in bitext:
    n = n + 1
    for f in f_sen:
      if f not in f_prob:
        f_prob[f] = {}
        sum_t[f] = 0.0
      total = 0.0
      for e in e_sen:
        if e not in e_prob:
          e_prob[e] = 1 / float(len(e_sen))
        else
          e_prob[e] += 1 / float(len(e_sen))
        if e not in f_prob[f]:
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

#print bitext


'''
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

dice = defaultdict(int)
for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
  dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
  if k % 5000 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")

for (f, e) in bitext:
  for (i, f_i) in enumerate(f): 
    for (j, e_j) in enumerate(e):
      if dice[(f_i,e_j)] >= opts.threshold:
        sys.stdout.write("%i-%i " % (i,j))
  sys.stdout.write("\n")
'''