from __future__ import division

from sklearn import cluster
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
import numpy

import constants



class Learner:
	### Initialization ###
	# Creates a new support vector machine.
	def __init__(self):
		self.scaler = preprocessing.StandardScaler()
		
		self.machine = svm.SVC()
		self.clustering = cluster.AgglomerativeClustering(n_clusters = 10, affinity = "precomputed", linkage = "average")
	
		self.predictedSimilarities = {}


	### SVM ###
	# Scales the data to ~N(0, 1), stores scaling information for later
	# reference, fits the model.
	def fitSVM(self, trainExamples, trainLabels):
		self.machine.fit(self.scaler.fit_transform(trainExamples), trainLabels)
	
	
	# Scales the data, generates predictions.
	def predictSVM(self, testExamples):
		return self.machine.predict(self.scaler.transform(testExamples))
	
	
	### Clustering ###
	# For each meaning, clusters all wordforms.
	def cluster(self, wordforms, testMeanings, testLanguages, extractor):
		predictedClusters = {}
		
		# For each meaning in test data, clustering is performed separately.
		for meaningIndex in testMeanings:
			# Performs clustering to predict cluster assignments for each
			# wordform.
			meaningLanguages = self.collectMeaningLanguages(testLanguages, wordforms[meaningIndex])
			distances = self.computeDistances(meaningLanguages, wordforms[meaningIndex], extractor)
			predictedAssignments = self.clustering.fit_predict(numpy.array(distances))
			
			# Formats the data so that it can be easier read and compared.
			predictedClusters[meaningIndex] = self.extractClusters(predictedAssignments, meaningLanguages, wordforms[meaningIndex])
		
		return predictedClusters
	
	
	# Depending on how the training and test sets were made, not all languages
	# might be represented within a meaning. Some wordforms for a meaning are
	# also simply missing in the original data. The method thus collects all
	# languages within a meaning that have a corresponding wordform.
	def collectMeaningLanguages(self, testLanguages, meaningWordforms):
		return [language for language in testLanguages if language in meaningWordforms]

	
	# Generates a matrix of all possible languages, with cell values set to the
	# distance between every two word pairs for the given meaning.
	def computeDistances(self, meaningLanguages, meaningWordforms, extractor):
		languageCount = len(meaningLanguages)
		
		distances = [[0] * languageCount for i in range(languageCount)]
		
		for i in range(languageCount):
			language1 = meaningLanguages[i]
			form1 = meaningWordforms.get(language1, None)
				
			if not form1:
				continue
		
			for j in range(languageCount):
				language2 = meaningLanguages[j]
				form2 = meaningWordforms.get(language2, None)
					
				if not form2:
					continue
			
				example = extractor(form1, form2, self.predictedSimilarities[language1][language2]) if self.predictedSimilarities else extractor(form1, form2)
				distances[i][j] = 1 - self.predictSVM(example)[0]
	
		return distances
	
	
	# Generates readable clusters from clustering output showing all wordforms
	# and their languages for each cluster.
	def extractClusters(self, predictedAssignments, meaningLanguages, meaningWordforms):
		clusters = {}
		
		for i, clusterIndex in enumerate(predictedAssignments):
			languageIndex = meaningLanguages[i]
			wordform = meaningWordforms[languageIndex]
				
			if clusterIndex not in clusters:
				clusters[clusterIndex] = []
			clusters[clusterIndex].append((meaningWordforms[languageIndex], languageIndex))
	
		return clusters


	### Evaluation ###
	# Computes accuracy of predictions by comaring them to the truth.
	def computeAccuracy(self, truth, predictions):
		return metrics.accuracy_score(truth, predictions)
	
	
	# Generates an evaluation report, where precision, recall and F-1 scores are
	# reported for each class separately, and for the entire dataset.
	def evaluatePairwise(self, truth, predictions):
		return metrics.classification_report(truth, predictions, target_names = constants.TARGETS)

	
	### Predicted Language Similarity ###
	# Uses the learner to generate cognatenes predictions for every possible
	# word pair for every meaning. Uses these predictions to compute predicted
	# language pair similarity as a ratio of positive predictions to all
	# predictions.
	def predictLanguageSimilarity(self, wordforms, extractor):
		predictedCounts = self.countPredictions(wordforms, extractor)
		self.computeSimilarity(predictedCounts)
		

	# Generates a cognateness decision for each wordform and meaning, counts the
	# number of positive and all predictions for each language pair.
	def countPredictions(self, wordforms, extractor):
		# Initializes the predicted counts dictionary so that it can later be
		# used without additional key existence checks.
		predictedCounts = {i: {j: [0, 0] for j in range(constants.LANGUAGE_COUNT + 1)} for i in range(constants.LANGUAGE_COUNT + 1)}
		
		# For each meaning, for each pair of languages, extracts features from
		# the two wordforms, uses the learner to predict cognateness.
		for meaningIndex in range(1, constants.MEANING_COUNT + 1):
			for language1 in range(1, constants.LANGUAGE_COUNT + 1):
				for language2 in range(1, constants.LANGUAGE_COUNT + 1):
					form1 = wordforms[meaningIndex].get(language1, None)
					form2 = wordforms[meaningIndex].get(language2, None)
					
					if not form1 or not form2:
						continue
				
					prediction = self.predictSVM(extractor(form1, form2))
					
					predictedCounts[language1][language2][0] += 1
					predictedCounts[language1][language2][1] += prediction

		return predictedCounts


	# Once all predictions are generated, computes predicted language pair
	# similarity using counts of positive and all predictions.
	def computeSimilarity(self, predictedCounts):
		predictedSims = {i: {j: 0 for j in range(constants.LANGUAGE_COUNT + 1)} for i in range(constants.LANGUAGE_COUNT + 1)}
		
		for language1 in range(1, constants.LANGUAGE_COUNT + 1):
			for language2 in range(1, constants.LANGUAGE_COUNT + 1):
				numerator = predictedCounts[language1][language2][1]
				denominator = predictedCounts[language1][language2][0]
				
				if denominator > 0:
					predictedSims[language1][language2] = numerator / denominator
				else:
					predictedSims[language1][language2] = 0.0
		
		self.predictedSimilarities = predictedSims