import math
from collections import defaultdict

class model1(object):

    def __init__(self, iterate):
      self.iter = iterate
      self.theta = defaultdict(int)
        
    #training IBM Model1
    def train(self, bitext):

        #Create vocabulary mapping
        vocab = defaultdict(set)
        for (f, e) in bitext:
            for e_1 in e:
                for f_1 in f:
                    vocab[e_1].add(f_1)

        #Initialize Theta Probablities
        for e in vocab:
            for f in vocab[e]:
                self.theta[(e, f)] = 1.0 / float(len(vocab[e]))

        for z in range(self.iter):
            count = defaultdict(float)
            total = defaultdict(float)

            for (f, e) in bitext:
                #normalization
                s_total = defaultdict(float)
                for e_1 in e:
                    for f_1 in f:
                        s_total[e_1] += self.theta[(e_1, f_1)]

                # collect counts
                for e_1 in e:
                    for f_1 in f:
                        count[(e_1, f_1)] += self.theta[(e_1, f_1)] / s_total[e_1]
                        total[f_1] += self.theta[(e_1, f_1)] / s_total[e_1]

            #calculate the probabilities
            for e in vocab:
                for f in vocab[e]:
                    self.theta[(e, f)] = count[(e, f)] / total[f]

    #Finding the best alignments
    def align(self, (f, e)):
        best_align = []
        for (j, e_j) in enumerate(e):
            best_prob = 0
            best_i = 0
            for (i, f_i) in enumerate(f):
                if self.theta[(e_j, f_i)] > best_prob:
                    best_prob = self.theta[(e_j, f_i)]
                    best_i = i
                elif self.theta[(e_j, f_i)] == best_prob and abs(j - i) < abs(j - best_i):
                    best_prob = self.theta[(e_j, f_i)]
                    best_i = i
            best_align.append((best_i, j))
                
        return best_align