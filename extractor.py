from __future__ import division
import os

from leven import levenshtein
import numpy

import constants



class Extractor:
	### Initialization ###
	# Initializes training and test set data structures.
	def __init__(self):
		# Measures
		self.identicalWordsMeasure = [self.identicalWords]
		self.identicalFirstLettersMeasure = [self.identicalFirstLetters]
		self.identicalPrefixesMeasure = [self.identicalPrefixes]
		self.MEDMeasure = [self.basicMED]
		self.HK2011Measures = [self.basicMED, self.LCPLength, self.commonBigramNumber, self.longerWordLen, self.shorterWordLen, self.wordLenDifference]
		
		self.trainExamples = []
		self.trainLabels = []

		self.testExamples = []
		self.testLabels = []
	
	
	### Pairwise Baselines ###
	# Identical words baseline (pairwise deduction).
	def identicalWordsBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, self.identicalWordsMeasure)

	
	# Identical first letter baseline (pairwise deduction).
	def identicalFirstLettersBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, self.identicalFirstLettersMeasure)
	
	
	# Identical prefix baseline (pairwise deduction).
	def identicalPrefixesBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, self.identicalPrefixesMeasure)
	
	
	# Minimum edit distance baseline (assumes costs of insertion, deletion and
	# substitution are all 1).
	def MEDBaseline(self, allExamples, allLabels):
		self.batchCompute(allExamples, allLabels, self.MEDMeasure)
	
	
	### Group-based Baselines ###
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
		self.batchCompute(allExamples, allLabels, self.HK2011Measures)
		self.addLanguageSimilarity(allExamples)
	
	
	# Arranges wordforms into groups of identical items.
	def identicalWordsGroupBaseline(self, testMeanings, testLanguages, wordforms):
		return self.groupBaseline(self.getWordform, testMeanings, testLanguages, wordforms)
	
	
	# Arranges wordforms into groups of items sharing the same first letter.
	def identicalFirstLettersGroupBaseline(self, testMeanings, testLanguages, wordforms):
		return self.groupBaseline(self.getFirstLetter, testMeanings, testLanguages, wordforms)
	
	
	# Arranges wordforms into groups of items sharing the first 4 letters (note
	# that if a word is shorter than 4 letters, it is automatically placed in
	# a separate cluster).
	def identicalPrefixesGroupBaseline(self, testMeanings, testLanguages, wordforms):
		return self.groupBaseline(self.getPrefix, testMeanings, testLanguages, wordforms)
	
	
	# Arranges wordforms for each meaning in groups of cognates, where a
	# cognateness decision is made based on the test method provided. The test
	# dataset is used.
	def groupBaseline(self, test, testMeanings, testLanguages, wordforms):
		clusters = {}
		
		for meaningIndex in testMeanings:
			clusterIndices = {}
			lastClusterIndex = -1

			clusters[meaningIndex] = {}
			
			for languageIndex, wordform in wordforms[meaningIndex].iteritems():
				key = test(wordform)
				
				# If the provided test returns None, the wordform is placed in
				# its own separate group.
				if key is None:
					lastClusterIndex += 1
					clusters[meaningIndex][lastClusterIndex] = [(wordform, languageIndex)]
				else:
					# clusterIndices stores keys provided by the test method and
					# their corresponding group numbers. This allows numbering
					# groups using consecutive numbers starting with zero.
					if key not in clusterIndices:
						lastClusterIndex += 1
						clusterIndices[key] = lastClusterIndex
				
					clusterIndex = clusterIndices[key]
					if clusterIndex not in clusters[meaningIndex]:
						clusters[meaningIndex][clusterIndex] = []
				
					clusters[meaningIndex][clusterIndex].append((wordform, languageIndex))
	
		labels = self.extractGroupLabels(clusters, wordforms, testMeanings, testLanguages)
		
		return labels, clusters
	
	
	### Extractors ###
	# Extracts a set of features (all used in Hauer & Kondrak) from a pair of
	# words.
	def HK2011Extractor(self, form1, form2, langSimilarity = None):
		return self.compute(form1, form2, self.HK2011Measures, langSimilarity)
	
	
	### Language Similarity ###
	# Extracts the necessary language similarity values from the language
	# similarity matrix, appends the new feature to the existing test set.
	def appendTestSimilarities(self, predictedSimilarities, allExamples):
		similarityFeature = []
		
		for i, (form1, form2, language1, language2) in enumerate(allExamples[constants.TEST]):
			similarityFeature.append(predictedSimilarities[language1][language2])
		
		self.testExamples = numpy.hstack((self.testExamples, numpy.array(similarityFeature)))
	
	
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


	### Feature Extraction ###
	# Uses the list of word similarity measures to generate a single example from
	# two wordforms.
	def compute(self, form1, form2, tests, langSimilarity):
		example = [test(form1, form2) for test in tests]
		
		if langSimilarity is not None:
			example.append(langSimilarity)
		
		return numpy.array(example)


	# Uses the provided test function to compare wordforms in each word pair and
	# assign a value based on the comparison. The computation is performed
	# separately for training and test sets unless deduction is set to True.
	# Since deduction involves no machine learning, the entire dataset is then
	# used as a test set.
	def batchCompute(self, allExamples, allLabels, tests):
		for purpose, examples in allExamples.iteritems():
			outLabels = allLabels[purpose]
			
			outExamples = []
			for i, (form1, form2, language1, language2) in enumerate(examples):
				testValues = []
				
				for test in tests:
					testValues.append(test(form1, form2))
				outExamples.append(testValues)
			
			if purpose == constants.TRAIN:
				self.trainExamples.extend(outExamples)
				self.trainLabels.extend(outLabels)
			elif purpose == constants.TEST:
				self.testExamples.extend(outExamples)
				self.testLabels.extend(outLabels)

		self.formatExamples()


	# Returns, for each meaning, a list of language-sorted cognate group label
	# indices for the test dataset.
	def extractGroupLabels(self, cognateSets, wordforms, testMeanings, testLanguages):
		groupLabels = {}
		
		for meaningIndex in testMeanings:
			labels = [-1] * len(wordforms[meaningIndex])
			keys = wordforms[meaningIndex].keys()
			
			for clusterIndex, entries in cognateSets[meaningIndex].iteritems():
				for (wordform, languageIndex) in entries:
					index = keys.index(languageIndex)
					labels[index] = clusterIndex
	
			groupLabels[meaningIndex] = []

			for index, label in enumerate(labels):
				languageIndex = keys[index]

				if (languageIndex in testLanguages):
					groupLabels[meaningIndex].append(label)

		return groupLabels


	### Word Similarity Measures ###
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
	
	
	# Computes minimum edit distance between the two wordforms. Here, all edit
	# operations have a cost of 1.
	def basicMED(self, form1, form2):
		return float(levenshtein(form1, form2))
	
	
	# Computes the length of the longest common prefix of the two wordforms.
	def LCPLength(self, form1, form2):
		return float(len(os.path.commonprefix([form1, form2])))
	
	
	# Computes the length of the longest common subsequence of the two
	# wordforms.
	def LCSLength(self, form1, form2):
		subsequence = self.LCS(form1, form2)
		return float(len(subsequence))
	
	
	# Computes Dice's coefficient based on shared bigrams.
	def bigramDice(self, form1, form2):
		return self.ngramDice(2, form1, form2)
	
	
	# Computes Dice's coefficient based on shared trigrams.
	def trigramDice(self, form1, form2):
		return self.ngramDice(3, form1, form2)
	
	
	# Computes Dice's coefficient based on n-grams of the two wordforms, where
	# s = 2z / x + y (s: similarity, z: number of shared n-grams, x: number of
	# n-grams in the first word, and y: number of n-grams in the second word.
	def ngramDice(self, n, form1, form2):
		if len(form1) < n or len(form2) < n:
			return 0.0
		else:
			return 2 * self.commonNgramNumber(n, form1, form2) / (len(form1) + len(form2) - 2 * (n - 1))

	
	# Computes the number of bigrams the two words share.
	def commonBigramNumber(self, form1, form2):
		return self.commonNgramNumber(2, form1, form2)
	
	
	# Computes the number of trigrams the two words share.
	def commonTrigramNumber(self, form1, form2):
		return self.commonNgramNumber(3, form1, form2)
	
	
	# Computes the number of n-grams the two words share.
	def commonNgramNumber(self, n, form1, form2):
		commonNgrams = self.commonNgrams(self.ngrams(n, form1), self.ngrams(n, form2))
		return float(len(commonNgrams))
	
	
	# Computes the ratio of shared bigrams of the two words.
	def commonBigramRatio(self, form1, form2):
		return self.commonNgramRatio(2, form1, form2)
	
	
	# Computes the ratio of shared trigrams of the two words.
	def commonTrigramRatio(self, form1, form2):
		return self.commonNgramRatio(3, form1, form2)


	# Computes the pair's shared n-gram ratio by dividing the number of shared
	# n-grams of the two wordforms by the number of n-grams in the longer word.
	def commonNgramRatio(self, n, form1, form2):
		return self.commonNgramNumber(n, form1,form2) / (self.longerWordLen(form1, form2) - (n - 1))
	
	
	# Computes the length of the longer of the two words.
	def longerWordLen(self, form1, form2):
		return float(len(form1)) if len(form1) > len(form2) else float(len(form2))
	
	
	# Computes the length of the shorter of the two words.
	def shorterWordLen(self, form1, form2):
		return float(len(form1)) if len(form1) < len(form2) else float(len(form2))
	
	
	# Computes the absolute difference between the lengths of the two words.
	def wordLenDifference(self, form1, form2):
		return float(abs(len(form1) - len(form2)))
	
	
	# Finds the longest common subsequence of the two wordforms.
	def LCS(self, form1, form2):
		if not form1 or not form2:
			return ""
		
		firstForm1, restForm1, firstForm2, restForm2 = form1[0], form1[1 :], form2[0], form2[1 :]
		if firstForm1 == firstForm2:
			return firstForm1 + self.LCS(restForm1, restForm2)
		else:
			return max(self.LCS(form1, restForm2), self.LCS(restForm1, form2), key = len)
	
	
	# Generates a list of the word's n-grams.
	def ngrams(self, n, word):
		return [word[i : i + n] for i in range(len(word) - n + 1)]
	
	
	# Given two n-gram lists, creates a single list that contains all common
	# ngrams.
	def commonNgrams(self, bigrams1, bigrams2):
		ngrams = []
	
		for bigram in bigrams1:
			if bigram in bigrams2:
				ngrams.append(bigram)
				bigrams2.remove(bigram)

		return ngrams
	
	
	### Baseline Tests ###
	# Returns the wordform itself.
	def getWordform(self, wordform):
		return wordform
	
	
	# Returns the first letter of the wordform.
	def getFirstLetter(self, wordform):
		return wordform[0]
	
	
	# Returns None for wordforms shorter than four characters, and the first
	# four characters of the wordform otherwise.
	def getPrefix(self, wordform):
		if len(wordform) < 4:
			return None
		else:
			return wordform[ : 4]


	### Formatting ###
	# Formats the output.
	def formatExamples(self):
		self.trainExamples = numpy.array(self.trainExamples)
		self.testExamples = numpy.array(self.testExamples)