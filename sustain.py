"""This is an implementation of SUSTAIN (Love, Medin, & Gureckis, 2004). The code here was forked and updated from Gureckis Lab's version on GitHub (https://github.com/NYUCCL/sustain_python).

This version improves on the previous version by accommodating continuous features and removing all the pesky functional programming.

This version is set up for classification with continuous, discrete, or a combination of feature types. It is not set up for inference. However, trivially simple modifications would be required to accommodate inference--replace the feature/category masks with ones that mask an arbitrary dimension and add arguments to 'stimulate' and 'learn' that informs which dimension to mask.

The model expects inputs to be a np.array with the class label occupying the last index of the array. An input for a two-feature stimulus might look like np.array([.4, .5, 1]) if in category B and np.array([.4, .5, 0]) if in category A.

Implementation by John D. Patterson, 2020

"""

import os, sys
import math
import numpy as np
import tempfile
from time import sleep
import string
from sets import *
from random import random, randint, shuffle


class SUSTAIN:
    def __init__(self, r, beta, d, tau, learningrate, initialphas, dimmaxes, dimmins, num_slots, dimtypes):
        # HYPERPARAMETERS
        self.R = r  # attention parameter; scalar value
        self.BETA = beta # cluster competition parameter; scalar value
        self.D = d # response mapping parameter; scalar value
        self.TAU = tau # cluster creation criterion; for unsupervised learning, set to 0.0 if supervised
        self.LR = learningrate # scalar value
        self.LAMBDAS = initialphas # initial dimensional attention weighting; should be numpy array

        # CONVENIENCE VALS
        self.dimmaxes = dimmaxes # np.array with max possible val for each dimension
        self.dimmins = dimmins # same as ^, but min
        self.dimtypes = dimtypes # string with D characters; c = continuous, d = discrete (eg 'ccd')
        self.num_slots = num_slots# the # levels per dimension [1,2,..n]; in case of continuous dimension, dim should be set to 1
        self.featuremask = np.append(np.zeros(dimmaxes.size-1, dtype = float), 1) # masks the features
        self.categorymask = np.append(np.ones(dimmaxes.size-1, dtype = float), 0) # masks the category
        self.featureranges = np.abs(dimmaxes-dimmins) # scaling variables (for distance metric)
        if 'd' in self.dimtypes: # set featureranges for nominal dimensions to 2
            nominds = np.array([i for i, char in enumerate(self.dimtypes) if char == 'd']) # find indices of nominal dims
            self.featureranges[nominds] = 2

        # REPRESENTATION STORAGE
        self.clusters = []
        self.clusterdistances = []
        self.clusteractivations = []
        self.clusterouts = []
        self.weights = []
        self.outacts = []

    ## Make one-hot vector to represent discrete feature/category
    def make_onehot(self, num_slots, idx):
        target = np.zeros((1,int(num_slots)),dtype = float)[0]
        target[int(idx)] = 1.0
        return target.astype(float)

    ## Convert item from flat vector to numpy object
    def convert_item(self, item):
        iobj = np.empty(len(item), dtype = object) # numpy obj to store each dimension's array
        idx = 0
        for dtype in self.dimtypes:
            if dtype == 'c': #if continuous, simply use 1D numpy array
                iobj[idx] = np.array([item[idx]], dtype = float)
            else:
                iobj[idx] = self.make_onehot(self.num_slots[idx], item[idx])
            idx += 1

        return iobj

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #     S  T  I  M  U  L  A  T  E
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def stimulate(self, item, actual_response = None):

        # Convert item into helpful numpy object format
        if item.dtype != object:
            item = self.convert_item(item)

        # Compute distance between clusters and input (eq. 4)
        self.clusterdistances = []
        for cluster in self.clusters:
            normalized_dist = np.true_divide(np.abs(item-cluster), self.featureranges)
            summed_dim_dist = np.array([np.sum(dim) for dim in normalized_dist]) # sum up distances in each dimension (in case d > 1)
            self.clusterdistances.append(summed_dim_dist)
        # self.clusterdistances = [np.true_divide(np.abs(item-cluster), self.featureranges, dtype = float) for cluster in self.clusters]

        # Attention-weighted cluster activations (eq. 5)
        lambdaRs = np.multiply(self.categorymask, np.power(self.LAMBDAS, self.R))
        sumlambdaRs = np.hstack(lambdaRs).sum()
        self.clusteractivations = []
        for distance in self.clusterdistances:
            self.clusteractivations.append((lambdaRs * np.exp(-1.0 * distance * self.LAMBDAS)).sum() / sumlambdaRs)

        # do cluster competition (most activated cluster; eq. 6) & get weighted output node activations (eq. 7)
        assert self.BETA >= 0, "Negative BETA; BETA should be positive"
        if len(self.clusteractivations) > 0: # if cluster activations

            # get most activated cluster (eq. 6)
            num = np.power(self.clusteractivations, self.BETA, dtype = float) # numerator of equation 6
            self.clusterouts = np.multiply(num/num.sum(), self.clusteractivations)
            activeindex = np.argmax(self.clusterouts) # get winning cluster index

            # association weights: calculate response/class node activations (eq. 7)
            self.outacts = self.clusterouts[activeindex] * self.weights[activeindex]

        else: # if no cluster activations, set response nodes to 0
            self.outacts = item.copy()*0

        # Luce-ish response probabilities (eq. 8)
        assert self.D >=0, "Negative D; D is always positive"
        self.probs = self.outacts.copy() # copy the numpy object to use as a frame for response probabilities
        for dim in range(self.outacts.size):
            if self.outacts[dim].size > 1: # for nominal units, do luce
                nomunit = self.outacts[dim]
                expsum = np.exp(self.D * nomunit).sum()
                for val in range(nomunit.size):
                    self.probs[dim][val] = np.exp(self.D * nomunit[val])/expsum
            if self.outacts[dim].size == 1: # for continuous units, just leave them be (maybe make t-dist)
                self.probs[dim] = self.outacts[dim]

        # Get response
        maskedprobs = self.probs * self.featuremask
        maskedprobsflat = np.hstack(maskedprobs)
        itemflat = np.hstack(item)
        probcorrect = max(maskedprobsflat * itemflat)

        if actual_response == None: # if NOT learning from actual behavioral decisions...
            # print('actual resp: ' + str(actual_response))
            # if probcorrect > 0.5:
            #     response = 1
            # else:
            #     response = 0
            randnum = random()
            # print('randnum: ' + str(randnum))
            # print('probcorrect: ' + str(probcorrect))
            if randnum > probcorrect:
                response = 0 # incorrect response
            else:
                response = 1 # correct response
        else: # if using ACTUAL behavioral responses to fit data, do this
            response = actual_response

        # Prepare return vars
        if len(self.clusters) > 0:
            outacts = np.hstack(self.outacts)
            clusteractivations = np.vstack(self.clusteractivations)
            clusterdistances = np.vstack(self.clusterdistances)
            clusterouts = np.hstack(self.clusterouts)
            if len(self.clusters) == 1:
                clusters = np.array(np.hstack(self.clusters[0]))
                # print('NON-stack clusters')
                # print(clusters)
            else:
                clusters = np.array(np.vstack([np.hstack(i) for i in self.clusters]))
                # print('stack clusters')
                # print(clusters)


        else:
            outacts = np.zeros_like(self.outacts)
            clusteractivations = np.zeros_like(self.clusteractivations)
            clusterdistances = np.zeros_like(self.clusterdistances)
            clusterouts = np.zeros_like(self.clusterouts)
            clusters = np.zeros_like(self.clusters)


        return [response, probcorrect, self.probs[-1], outacts, clusteractivations, clusterdistances, clusterouts, self.LAMBDAS, clusters]

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #            L  E  A  R  N
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def learn(self, item, actual_response = None):
        # print('\n\n---LEARNING METHOD---')
        if len(self.clusters) == 0: # if no clusters yet, recruit item as cluster and train weights
            if item.dtype != object:
                cluster = self.convert_item(item)
            else:
                cluster = item.copy()
            self.clusters.append(cluster)
            self.weights.append(cluster * 0) # takes the slot frame to serve as weights holder
            self.stimulate(item, actual_response = actual_response)
            activeindex = np.argmax(self.clusteractivations)
            self.adjustcluster(activeindex, item)
            clusters = np.hstack(self.clusters)

        else: # otherwise, determine if most active cluster is associated with correct category (eq. 10)
            if item.dtype != object:
                item = self.convert_item(item)
            activeindex = np.argmax(self.clusteractivations)
            maskeditem = item * self.featuremask
            maskedcluster = self.clusters[activeindex] * self.featuremask
            maskeddist = np.hstack(np.divide(np.abs(maskeditem-maskedcluster), self.featureranges)).sum()

            if (max(self.clusteractivations) < self.TAU) or (maskeddist != 0.0):
                # recruit new cluster
                if item.dtype != object:
                    cluster = self.convert_item(item)
                else:
                    cluster = item
                self.clusters.append(cluster)
                self.weights.append(cluster * 0) # takes the slot frame to serve as weights holder
                self.stimulate(item, actual_response = actual_response)
                activeindex = np.argmax(self.clusteractivations)
                self.adjustcluster(activeindex, item)
            else:
                self.adjustcluster(activeindex, item)

            if len(self.clusters) > 1:
                clusters = np.vstack([np.hstack(i) for i in self.clusters])
            else:
                clusters = np.hstack(self.clusters[0])
        return [self.LAMBDAS, self.weights, clusters]

    # define humble teacher (eq. 9)
    def humbleteacher(self, item):
        deltas = item * 0   # copy item to use as shell for getting teacher values
        for dim in range(item.size): # do a for loop thru the different dimensions and slots within each dimension
            #  for discrete/binomial dimensions
            if item[dim].size > 1:
                for val in range(item[dim].size):
                    if (((self.outacts[dim][val] > self.dimmaxes[dim]) and (item[dim][val] == self.dimmaxes[dim])) or \
                        ((self.outacts[dim][val] < self.dimmins[dim]) and (item[dim][val] == self.dimmins[dim]))):
                        deltas[dim][val] = 0.0  # if the predicted (outacts) goes above and beyond objective, set delta to 0
                    else:
                        deltas[dim][val] = item[dim][val] - self.outacts[dim][val]  # if not, set delta to the delta
            #  for continuous dimensions
            else:
                if (((self.outacts[dim] > self.dimmaxes[dim]) and (item[dim] == self.dimmaxes[dim])) or \
                    ((self.outacts[dim] < self.dimmins[dim]) and (item[dim] == self.dimmins[dim]))):
                    deltas[dim] = 0.0 # if the predicted (outacts) goes above and beyond objective, set delta to 0
                else:
                    deltas[dim] = item[dim] - self.outacts[dim]  # if not, set delta to the delta

        return deltas

    def adjustcluster(self, activeindex, item):
        if item.dtype != object:
            item = self.convert_item(item)
        deltas = self.humbleteacher(item)  # compute errors

        # mask non-queried dimensions & update weights (eq. 14)
        deltas = deltas * self.featuremask
        # print('out deltas')
        # print(deltas)
        self.weights[activeindex] += self.LR * deltas * self.clusterouts[activeindex]


        # update cluster positions (eq. 12)
        deltas = item - self.clusters[activeindex]
        # print('clust deltas')
        # print(deltas)
        # print('clust before')
        # print(self.clusters[activeindex])
        self.clusters[activeindex] = self.clusters[activeindex] + (self.LR * deltas)
        # print('clust after')
        # print(self.clusters[activeindex])


        # update lambdas/receptive fields tunings (eq. 13)
        lambdaMus = self.clusterdistances[activeindex] * self.LAMBDAS
        expLambdaMus = lambdaMus.copy() * 0
        for dim in range(lambdaMus.size):
            expLambdaMus[dim] = np.exp(-1.0 * lambdaMus[dim], dtype = float)

        self.LAMBDAS += self.LR * expLambdaMus * (1.0 - lambdaMus)


def main(): #  Example usage (I havent tested this, but it probably works)
    num_blocks = 5
    num_inits = 20
    prob = [ [0,0,0,0], # SHJ type 1
             [0,0,1,0],
             [0,1,0,0],
             [0,1,1,0],
             [1,0,0,1],
             [1,0,1,1],
             [1,1,0,1],
             [1,1,1,1] ]
    initdata = np.zeros((num_inits, num_blocks))
    for init in range(num_inits):
        ## Initialize model
        model = SUSTAIN(r = 9.01245,
                        beta = 1.252233,
                        d = 16.924073,
                        tau = 0.0,
                        learningrate = 0.1,
                        initialphas = np.ones(4),
                        dimmaxes = np.array([1, 1, 1, 1]),
                        dimmins= np.array([0, 0, 0, 0]),
                        num_slots=np.array([2, 2, 2, 2]),
                        dimtypes='dddd')

        blockdata = np.zeros(num_blocks) # init block data storage

        ##  Training Loop
        for block in range(num_blocks):
            shuffle(prob)

            trialdata = np.zeros(len(prob)) # init trial data storage

            for trial in range(len(prob)):

                item = np.array(prob[trial])

                # forward pass step
                [response,probcorrect,probabilities,outacts,clusteractivations,clusterdistances, \
                 clusterouts, lambdas, clusters] = model.stimulate(item = item)
                # print('repsonse')
                # print(response)

                trialdata[trial] = probcorrect

                # learn step
                [lambdas,weights,clusters] = model.learn(item = item)

            blockdata[block] = trialdata.mean()

        initdata[init] = blockdata
    print(initdata.mean(axis=0))

if __name__ == '__main__':
    main()
