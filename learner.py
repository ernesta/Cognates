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
		self.machine = svm.LinearSVC(verbose = True, fit_intercept = False)
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
	# For each meaning, clusters all wordforms in the test dataset.
	def cluster(self, wordforms, testMeanings, testLanguages, extractor):
		predictedLabels = {}
		predictedClusters = {}
		clusterCounts = {}
		clusterDistances = {}
		
		for meaningIndex in testMeanings:
			meaningLanguages = self.collectMeaningLanguages(testLanguages, wordforms[meaningIndex])
			distances = self.computeDistances(meaningLanguages, wordforms[meaningIndex], extractor)
			
			for n in range(constants.CLUSTER_MIN, constants.CLUSTER_MAX + 1):
				# Clusters the data into n groups.
				clustering = cluster.AgglomerativeClustering(n_clusters = n, affinity = "precomputed", linkage = "average")
				labels = clustering.fit_predict(numpy.array(distances))
			
				# Finds the smallest distance between clusters.
				minDistance = self.computeMinClusterDistance(n, distances, labels)
				
				if minDistance <= constants.THRESHOLD:
					predictedLabels[meaningIndex] = labels
					predictedClusters[meaningIndex] = self.extractClusters(labels, meaningLanguages, wordforms[meaningIndex])
					
					clusterCounts[meaningIndex] = n
					clusterDistances[meaningIndex] = minDistance
					
					break

		return predictedLabels, predictedClusters, clusterCounts, clusterDistances
	
	
	# Computes the optimal cluster distance threshold for clustering.
	def computeDistanceThreshold(self, wordforms, testMeanings, testLanguages, extractor, trueLabels):
		sumDistances = 0.0
		
		for meaningIndex in testMeanings:
			V1scores = []
			minDistances = []
			
			meaningLanguages = self.collectMeaningLanguages(testLanguages, wordforms[meaningIndex])
			distances = self.computeDistances(meaningLanguages, wordforms[meaningIndex], extractor)
	
			for n in range(constants.CLUSTER_MIN, constants.CLUSTER_MAX + 1):
				clustering = cluster.AgglomerativeClustering(n_clusters = n, affinity = "precomputed", linkage = "average")
				labels = clustering.fit_predict(numpy.array(distances))
				
				# Finds the smallest distance between clusters.
				minDistance = self.computeMinClusterDistance(n, distances, labels)
				V1 = self.computeV1(trueLabels[meaningIndex], labels)
				
				print meaningIndex, n, V1, minDistance
			
				minDistances.append(minDistance)
				V1scores.append(V1)
			
			print "\n"
				
			maxV1 = max(V1scores)
			index = V1scores.index(maxV1)
			sumDistances += minDistances[index]
	
		return sumDistances / len(testMeanings)
	
	
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
	
	
	# Given cluster assignments and distances between each wordform of a
	# meaning, computes average cluster distances and returns the smallest
	# one.
	def computeMinClusterDistance(self, n, distances, predictedLabels):
		clusterSums = [[0] * n for i in range(n)]
		clusterCounts = [[0] * n for i in range(n)]

		for i in range(len(distances)):
			iLabel = predictedLabels[i]

			for j in range(i, len(distances)):
				jLabel = predictedLabels[j]

				distance = distances[i][j]

				clusterSums[iLabel][jLabel] += distance
				clusterCounts[iLabel][jLabel] += 1
					
				clusterSums[jLabel][iLabel] += distance
				clusterCounts[jLabel][iLabel] += 1
	
		clusterDistances = numpy.array(clusterSums) / numpy.array(clusterCounts)
		numpy.fill_diagonal(clusterDistances, 1.0)

		return clusterDistances.min()
	
	
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
	# Computes accuracy of predictions by comparing them to the truth.
	def computeAccuracy(self, truth, predictions):
		return metrics.accuracy_score(truth, predictions)
	
	
	# Generates an evaluation report, where precision, recall and F-1 scores are
	# reported for each class separately, and for the entire dataset.
	def evaluatePairwise(self, truth, predictions):
		return metrics.classification_report(truth, predictions, target_names = constants.TARGETS)
	
	
	# Computes the V1 score of the predicted grouping of wordforms for a meaning
	# compared to the actual cognate grouping.
	def computeV1(self, truth, predictions):
		return metrics.v_measure_score(truth, predictions)

	
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