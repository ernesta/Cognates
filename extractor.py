from __future__ import division
import os

from leven import levenshtein
import numpy

import constants



class Extractor:
	# Initializes training and test set data structures.
	def __init__(self):
		# Metrics
		self.HK2011Metrics = [self.MED, self.LCPLength, self.commonBigramNumber, self.longerWordLen, self.shorterWordLen, self.wordLenDifference]
		
		self.trainExamples = []
		self.trainLabels = []

		self.testExamples = []
		self.testLabels = []
	
	
	# Identical words baseline (pairwise deduction).
	def identicalWordsBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, [self.identicalWords], True)

	
	# Identical first letter baseline (pairwise deduction).
	def identicalFirstLettersBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, [self.identicalFirstLetters], True)
	
	
	# Identical prefix baseline (pairwise deduction).
	def identicalPrefixesBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, [self.identicalPrefixes], True)
	
	
	# Minimum edit distance baseline (assumes costs of insertion, deletion and
	# substitution are all 1).
	def MEDBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, [self.MED])
	
	
	# A reproduction of Hauer & Kondrak (2011). Since data has been preprocessed
	# slightly differently in this project, and in some cases Hauer & Kondrak
	# only provide minimal implementation information, a *true* comparison among
	# the different approaches can only be performed if the Hauer & Kondrak
	# approach is re-implemented.
	def HK2011Baseline(self, allExamples, allLabels):
		# Minimum edit distance (here used with all costs equal to 1)
		# Longest common prefix length
		# Number of common bigrams
		# Length of the first word (here length of the shorter word)
		# Length of the second word (here length of the longer word)
		# Absolute length difference between the two words
		self.batchCompute(allExamples, allLabels, self.HK2011Metrics)
		self.addLanguageSimilarity(allExamples)
	
	
	# Extracts a set of features (all used in Hauer & Kondrak) from a pair of
	# words.
	def HK2011Extractor(self, form1, form2, langSimilarity = None):
		return self.compute(form1, form2, self.HK2011Metrics, langSimilarity)
	
	
	# Extracts the necessary language similarity values from the language
	# similarity matrix, appends the new feature to the existing test set.
	def appendTestSimilarities(self, predictedSimilarities, allExamples):
		similarityFeature = []
		
		for i, (form1, form2, language1, language2) in enumerate(allExamples[constants.TEST]):
			similarityFeature.append(predictedSimilarities[language1][language2])
		
		self.testExamples = numpy.hstack((self.testExamples, numpy.array(similarityFeature)))

	
	# Uses the provided test function to compare wordforms in each word pair and
	# assign a value based on the comparison. The computation is performed
	# separately for training and test sets unless deduction is set to True.
	# Since deduction involves no machine learning, the entire dataset is then
	# used as a test set.
	def batchCompute(self, allExamples, allLabels, tests, deduction = False):
		for purpose, examples in allExamples.iteritems():
			outLabels = allLabels[purpose]
			
			outExamples = []
			for i, (form1, form2, language1, language2) in enumerate(examples):
				testValues = []
				
				for test in tests:
					testValues.append(test(form1, form2))
				outExamples.append(testValues)

			purpose = constants.TEST if deduction else purpose
			
			if purpose == constants.TRAIN:
				self.trainExamples.extend(outExamples)
				self.trainLabels.extend(outLabels)
			elif purpose == constants.TEST:
				self.testExamples.extend(outExamples)
				self.testLabels.extend(outLabels)

		self.formatExamples()


	# Uses the list of word similarity metrics to generate a single example from
	# two wordforms.
	def compute(self, form1, form2, tests, langSimilarity):
		example = [test(form1, form2) for test in tests]
		
		if langSimilarity is not None:
			example.append(langSimilarity)
		
		return numpy.array(example)


	# Measures language similarity as a fraction of positive examples to all
	# examples for each language pair in the training set.
	def addLanguageSimilarity(self, allExamples):
		decisionCounts = self.countTrainDecisions(allExamples)
		decisionSimilarities = self.computeTrainSimilarity(allExamples, decisionCounts)
		
		self.trainExamples = numpy.column_stack((self.trainExamples, numpy.array(decisionSimilarities)))


	# Uses the training dataset to count positive and all cognateness decisions
	# for language pairs present in the data.
	def countTrainDecisions(self, allExamples):
		decisionCounts = {}
		
		for i, (form1, form2, language1, language2) in enumerate(allExamples[constants.TRAIN]):
			if language1 not in decisionCounts:
				decisionCounts[language1] = {}
			if language2 not in decisionCounts[language1]:
				decisionCounts[language1][language2] = [0, 0]
			
			decisionCounts[language1][language2][0] += 1
			
			if self.trainLabels[i] == 1:
				decisionCounts[language1][language2][1] += 1

		return decisionCounts
	
	
	# Once all decisions are counted, computes decision-based language pair
	# similarity using counts of positive and all decisions.
	def computeTrainSimilarity(self, allExamples, decisionCounts):
		decisionSimilarities = []
		
		for i, (form1, form2, language1, language2) in enumerate(allExamples[constants.TRAIN]):
			decisionSimilarities.append(decisionCounts[language1][language2][1] / decisionCounts[language1][language2][0])
	
		return decisionSimilarities


	# Checks if the two wordforms are identical.
	def identicalWords(self, form1, form2):
		return float(form1 == form2)

	
	# Checks if the two wordforms have the same first letter.
	def identicalFirstLetters(self, form1, form2):
		return float(form1[0] == form2[0])
	
	
	# Checks if the two wordforms have an identical prefix that is at least 4
	# characters long.
	def identicalPrefixes(self, form1, form2):
		return float(self.LCPLength(form1, form2) > 3)
	
	
	# Computes minimum edit distance between the two wordforms.
	def MED(self, form1, form2):
		return float(levenshtein(form1, form2))
	
	
	# Computes the length of the longest common prefix of the two wordforms.
	def LCPLength(self, form1, form2):
		return float(len(os.path.commonprefix([form1, form2])))
	
	
	# Computes the number of bigrams the two words share.
	def commonBigramNumber(self, form1, form2):
		bigrams1 = self.ngrams(2, form1)
		bigrams2 = self.ngrams(2, form2)
	
		commonBigrams = self.commonNgrams(bigrams1, bigrams2)
		return float(len(commonBigrams))
	
	
	# Computes the length of the longer of the two words.
	def longerWordLen(self, form1, form2):
		return float(len(form1)) if len(form1) > len(form2) else float(len(form2))
	
	
	# Computes the length of the shorter of the two words.
	def shorterWordLen(self, form1, form2):
		return float(len(form1)) if len(form1) < len(form2) else float(len(form2))
	
	
	# Computes the absolute difference between the lengths of the two words.
	def wordLenDifference(self, form1, form2):
		return float(abs(len(form1) - len(form2)))
	
	
	# Generates a list of the word's n-grams.
	def ngrams(self, n, word):
		return [word[i : i + n] for i in range(len(word) - n + 1)]
	
	
	# Given two ngram lists, creates a single list that contains all common
	# ngrams.
	def commonNgrams(self, bigrams1, bigrams2):
		ngrams = []
	
		for bigram in bigrams1:
			if bigram in bigrams2:
				ngrams.append(bigram)
				bigrams2.remove(bigram)

		return ngrams
	
	
	# Formats the output to adhere to scikit-learn requirements.
	def formatExamples(self):
		self.trainExamples = numpy.array(self.trainExamples)
		self.testExamples = numpy.array(self.testExamples)