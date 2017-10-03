import optparse
import sys
from collections import defaultdict

def IBM1(bitext):
  sys.stderr.write("Training with IBM Model 1 ...")
  f_count = defaultdict(int)
  e_count = defaultdict(int)
  fe_count = defaultdict(int)
  for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
      f_count[f_i] += 1
      for e_j in set(e):
        fe_count[(f_i,e_j)] += 1
    for e_j in set(e):
      e_count[e_j] += 1
    if n % 500 == 0:
      sys.stderr.write(".")

  max_iters = 5
  thres = 0.001
  
  #initialize to uniform distribution
  lf = len(f_count)
  trans_probs = defaultdict(float)
  for (f,e) in bitext:
    for e_i in set(e):
      for f_i in set(f):
        trans_probs[(e_i, f_i)] = 1.0/lf
        #sys.stderr.write(str(trans_probs[(e_i,f_i)]))

  for iteration in xrange(max_iters):
    sys.stderr.write('iteration: ' + str(iteration) + '\n')
    #initialization
    count = defaultdict(int)
    total = defaultdict(int)
    s_total = defaultdict(float)
    for(f, e) in bitext:
      #normalization
      for e_i in set(e):
        for f_i in set(f):
          s_total[e_i] += trans_probs[(e_i, f_i)]
      #counts
      for e_i in set(e):
        for f_i in set(f):
          count[(e_i,f_i)] += trans_probs[(e_i, f_i)]/s_total[e_i]
          total[f_i] += trans_probs[(e_i, f_i)]/s_total[e_i]
    #estimate probabilities
    for (e_i, f_i) in set(trans_probs.keys()):
        trans_probs[(e_i,f_i)] = float(count[(e_i,f_i)])/total[f_i]

  return trans_probs

def IBM2(bitext, trans_probs):
  sys.stderr.write("Training with IBM Model 2 ...")
  max_iters = 5
  thres = 0.001
  
  #initialize a to uniform distribution
  #make sure the trans_probs dist carries over from IBM1
  alignments = defaultdict(float)
  for (f,e) in bitext:
      le = len(e)
      lf = len(f)
      for (i, f_i) in enumerate(f):
        for (j, e_j) in enumerate(e):
          alignments[(i,j,le,lf)] = 1.0/(lf+1.0)

  for iteration in xrange(max_iters):
    sys.stderr.write('iteration: ' + str(iteration) + '\n')
    #initialization
    count = defaultdict(float)
    total = defaultdict(float)
    count_a = defaultdict(float)
    total_a = defaultdict(float)
    s_total = defaultdict(float)
    for(f, e) in bitext:
      le = len(e)
      lf = len(f)
      #normalization
      for (j, e_j) in enumerate(e):
        s_total[e_j] = 0
        for (i, f_i) in enumerate(f):
          s_total[e_j] += trans_probs[(e_j, f_i)]*alignments[(i,j,le,lf)]
      #counts
      for (j, e_j) in enumerate(e):
        for (i, f_i) in enumerate(f):
          c = trans_probs[(e_j, f_i)]*alignments[(i,j,le,lf)]/s_total[e_j]
          count[(e_j,f_i)] += c
          total[f_i] += c
          count_a[(i,j,le,lf)] += c
          total_a[(j,le,lf)] += c
    #estimate probabilities
    for (e_i, f_i) in set(trans_probs.keys()):
        trans_probs[(e_i,f_i)] = count[(e_i,f_i)]/total[f_i]
    for (f,e) in bitext:
      le = len(e)
      lf = len(f)
      for (i, f_i) in enumerate(f):
        for (j, e_j) in enumerate(e):
          alignments[(i,j,le,lf)] = count_a[(i,j,le,lf)] / total_a[(j,le,lf)]

  return trans_probs

if __name__ == '__main__':

  optparser = optparse.OptionParser()
  optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
  optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
  optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
  optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
  optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
  (opts, _) = optparser.parse_args()
  f_data = "%s.%s" % (opts.train, opts.french)
  e_data = "%s.%s" % (opts.train, opts.english)

  bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
  trans_probs1 = IBM1(bitext)
  trans_probs = IBM2(bitext, trans_probs1)
  #now in reverse
  bitext2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))[:opts.num_sents]]
  trans_probs1_r = IBM1(bitext2)
  trans_probs_r = IBM2(bitext2, trans_probs1_r)

  #generate alignments
  sys.stderr.write('generating alignments\n')
  for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
      e_max = -1
      e_max_r = -1
      e_max_p = 0
      e_max_p_r = 0
      for (j, e_j) in enumerate(e):
        if trans_probs[(e_j, f_i)] > e_max_p:
          e_max = j
          e_max_p = trans_probs[(e_j, f_i)]
        if trans_probs_r[(f_i, e_j)] > e_max_p_r:
          e_max_r = j
          e_max_p_r = trans_probs_r[(f_i, e_j)]
      if e_max == e_max_r:
        sys.stdout.write("%i-%i " % (i,e_max))
    sys.stdout.write("\n")