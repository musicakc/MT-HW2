import sys
import math
from collections import defaultdict

class model2(object):
    def __init__(self, iterate):
        self.pt = defaultdict(int)
        self.tt = defaultdict(lambda: 1.0)
        self.iterate = iterate 
        
    # Train HMM model
    def train(self, bitext):
        self.initialization(bitext)
        self.refinement(bitext)

    # Train transition probabilities
    def refinement(self, bitext):

        #Initialize tt
        for I in range(1, max([len(f) for (f, e) in bitext])):
            for j in range(I):
                for i in range(I):
                    self.tt[(i, j, I)] = 1.0 / I

        for k in range(self.iterate):
            s = defaultdict(int)

            for (f, e) in bitext:
                a = self.align((f, e))
                for i in range(len(a) - 1):
                    s[a[i][0] - a[i+1][0]] += 1.0
            
            #Estimate transition probabilities
            for p in range(1, max([len(f) for (f, e) in bitext])):
                for j in range(p):
                    total = sum([s[k - j] for k in range(p)])
                    for i in range(p):
                        self.tt[(i, j, p)] = s[i - j] / total
        
    # train translation probabilities
    def initialization(self, bitext):

        # create a mapping of possible (e, f) pairs
        vocab = defaultdict(set)
        for (f, e) in bitext:
            for e_j in e:
                for f_i in f:
                    vocab[e_j].add(f_i)

        # initialize t uniformly across possible pairs
        for e in vocab:
            for f in vocab[e]:
                self.pt[(e, f)] = 1.0 / float(len(vocab[e]))

        for k in range(self.iterate):

            # initialize
            count = defaultdict(float)
            total = defaultdict(float)

            for (f, e) in bitext:

                # compute normalization
                s_total = defaultdict(float)
                for e_i in e:
                    for f_j in f:
                        s_total[e_i] += self.pt[(e_i, f_j)]

                # collect counts
                for e_i in e:
                    for f_j in f:
                        count[(e_i, f_j)] += self.pt[(e_i, f_j)] / s_total[e_i]
                        total[f_j] += self.pt[(e_i, f_j)] / s_total[e_i]

            # estimate probabilities
            for e in vocab:
                for f in vocab[e]:
                    self.pt[(e, f)] = count[(e, f)] / total[f]

    #Return an alignment of (f, e) based on most likely approximation
    # k(j, i) = p(e_j | f_i) max [p(i | i', I) * k(j -1 , i')] 
    def align(self, (f, e)):
        
        #Initialize K and backpointers
        bp = {}
        K = defaultdict(int)

        # calculate K(0, i)
        for (i, f_i) in enumerate(f):
            K[(0, i)] = self.pt[(e[0], f_i)] 

        # calculate Q(j, i) and backpointers
        for (j, e_j) in enumerate(e[1:], 1):
            for (i, f_i) in enumerate(f):
                best_prob = -1
                for k in range(len(f)):
                    Kprime = self.tt[(i, k, len(f))] * K[(j - 1, k)]
                    if Kprime > best_prob:
                        K[(j, i)] = self.pt[(e_j, f_i)] * Kprime
                        bp[(j, i)] = (j - 1, k)
                        best_prob = Kprime
                    elif Kprime == best_prob and abs(j - 1 - k) < abs(j - 1 - bp[(j, i)][1]):
                        bp[(j, i)] = (j - 1, k)

        #Add the last alignment
        align_1 = []
        j, best = len(e) - 1, 0
        for i in range(len(f)):
            if K[(j, i)] >= K[(j, i)]:
                best = i
        align_1.append((i, j))

        #Use backpointers to generate rest of alignment
        while not j == 0:
            j, i = bp[(j, i)]
            align_1.append((i, j))

        return align_1