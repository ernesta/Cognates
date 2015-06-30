from __future__ import division
import math
import random

from sklearn import cluster
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
import numpy

import constants



class Learner:
	### Initialization ###
	# Initializes the standard scaler.
	def __init__(self):
		self.scaler = preprocessing.StandardScaler()
		self.predictedSimilarities = {}


	### SVM ###
	# Initializes SVM with a custom C value.
	def initSVM(self, C):
		self.SVM = svm.LinearSVC(C = C, fit_intercept = False, verbose = True)
	
	
	# Scales the data to ~N(0, 1), stores scaling information for later
	# reference, fits the SVM model.
	def fitSVM(self, trainExamples, trainLabels):
		self.SVM.fit(self.scaler.fit_transform(trainExamples), trainLabels)
	
	
	# Scales the data, generates SVM predictions.
	def predictSVM(self, testExamples):
		return self.SVM.predict(self.scaler.transform(testExamples))
	
	
	### Logistic Regression ###
	# Initializes logistic regression.
	def initLogisticRegression(self, C):
		self.LR = linear_model.LogisticRegression(C = C, fit_intercept = False, verbose = False)
	
	
	# Scales the data to ~N(0, 1), stores scaling information for later
	# reference, fits the linear regression.
	def fitLogisticRegression(self, trainExamples, trainLabels):
		self.LR.fit(self.scaler.fit_transform(trainExamples), trainLabels)
	
	
	# Scales the data, generates linear regression class predictions.
	def predictLogisticRegression(self, testExamples):
		return self.LR.predict(self.scaler.transform(testExamples))
	
	
	# Scales the data, generates linear regression probability predictions.
	def predictProbLogisticRegression(self, testExamples):
		return self.LR.predict_proba(self.scaler.transform(testExamples))[1]
	
	
	### Decision Tree Forest ###
	# Initializes a forest of trees. Equivalent to employing a gardener and
	# telling them how many trees and where to plant.
	def initForest(self, estimatorCount, seed):
		self.forest = ensemble.ExtraTreesClassifier(n_estimators = estimatorCount, random_state = seed)
	
	
	# Scales the data, trains the forest of randomized trees.
	def fitForest(self, trainExamples, trainLabels):
		self.forest.fit(self.scaler.fit_transform(trainExamples), trainLabels)
	
	
	# Returns features sorted by importance, and their importance values.
	def getForestImportances(self):
		importances = self.forest.feature_importances_
		indices = numpy.argsort(importances)[ : : -1]
	
		return importances, indices
	
	
	### Clustering ###
	# For each meaning, clusters all wordforms in the test dataset.
	def cluster(self, model, threshold, wordforms, testMeanings, testLanguages, extractor):
		predictedLabels = {}
		predictedClusters = {}
		clusterCounts = {}
		clusterDistances = {}
		
		for meaningIndex in testMeanings:
			meaningLanguages = self.collectMeaningLanguages(testLanguages, wordforms[meaningIndex])
			distances = self.computeDistances(model, meaningLanguages, testLanguages, wordforms[meaningIndex], extractor)
			
			for n in range(constants.CLUSTER_MIN, constants.CLUSTER_MAX + 1):
				# Clusters the data into n groups.
				clustering = cluster.AgglomerativeClustering(n_clusters = n, affinity = "precomputed", linkage = "average")
				labels = clustering.fit_predict(numpy.array(distances))
			
				# Finds the smallest distance between clusters.
				minDistance = self.computeMinClusterDistance(n, distances, labels)
				
				if minDistance <= threshold:
					predictedLabels[meaningIndex] = labels
					predictedClusters[meaningIndex] = self.extractClusters(labels, meaningLanguages, wordforms[meaningIndex])
					
					clusterCounts[meaningIndex] = n
					clusterDistances[meaningIndex] = minDistance
					
					break

		return predictedLabels, predictedClusters, clusterCounts, clusterDistances
	
	
	# Computes the optimal cluster distance threshold for clustering.
	def computeDistanceThreshold(self, model, wordforms, testMeanings, testLanguages, extractor, trueLabels):
		sumDistances = 0.0
		
		for meaningIndex in testMeanings:
			V1scores = []
			minDistances = []
			
			meaningLanguages = self.collectMeaningLanguages(testLanguages, wordforms[meaningIndex])
			distances = self.computeDistances(model, meaningLanguages, testLanguages, wordforms[meaningIndex], extractor)
	
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
	def computeDistances(self, model, meaningLanguages, testLanguages, meaningWordforms, extractor):
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
				
				example = extractor(form1, form2, testLanguages, language1, language2)

				if model == constants.SVM:
					distances[i][j] = 1 - self.predictSVM(example)[0]
				elif model == constants.LR:
					distances[i][j] = 1 - self.predictProbLinearRegression(example)[0]
	
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
	
	
	# Computes precision.
	def computeRecall(self, truth, predictions):
		return metrics.recall_score(truth, predictions)
	
	
	# Computes recall.
	def computePrecision(self, truth, predictions):
		return metrics.precision_score(truth, predictions)


	# Computes F1 of the positive class by comparing predictions to the truth.
	def computeF1(self, truth, predictions):
		return metrics.f1_score(truth, predictions)
	
	
	# Computes homogeneity of a clustering.
	def computeHomogeneity(self, truth, predictions):
		return metrics.homogeneity_score(truth, predictions)
	
	
	# Computes completeness of a clustering.
	def computeCompleteness(self, truth, predictions):
		return metrics.completeness_score(truth, predictions)
	
	
	# Computes the V1 score of the predicted grouping of wordforms for a meaning
	# compared to the actual cognate grouping.
	def computeV1(self, truth, predictions):
		return metrics.v_measure_score(truth, predictions)
	
	
	# Generates an evaluation report, where precision, recall and F-1 scores are
	# reported for each class separately, and for the entire dataset.
	def evaluatePairwise(self, truth, predictions):
		return metrics.classification_report(truth, predictions, target_names = constants.TARGETS)
	
	
	# Checks if a difference of performance between two clasifiers is
	# significant using the McNemar's test. Since we are interested in F1,
	# rather than accuracy significance, only predictions that pertain to the
	# classification of cognates is considered. The results are significant at
	# p = 0.05 if chi^2 > 3.841, and at p = 0.005 if chi^2 > 7.879.
	def computeMcNemarSignificance(self, truth, predictions1, predictions2):
		condition = (truth == 1)
		truth = numpy.extract(condition, truth)
		predictions1 = numpy.extract(condition, predictions1)
		predictions2 = numpy.extract(condition, predictions2)
	
		evals1 = (predictions1 == truth)
		evals2 = (predictions2 == truth)
		
		# Misclassified by the first model only: c01.
		# Misclassified by the second model only: c10.
		c01, c10 = 0, 0
	
		for i, eval1 in enumerate(evals1):
			eval2 = evals2[i]
			if eval1 == 0 and eval2 == 1:
				c01 += 1
			if eval1 == 1 and eval2 == 0:
				c10 += 1
		
		if c01 + c10 < 20:
			print "Unreliable conclusion:", c01, c10
		
		return math.pow(abs(c01 - c10) - 1, 2) / (c01 + c10)
	
	
	# Computes the probability that predictions of the two models are truly
	# different, and thus the resulting F1 scores indeed indicate significant
	# difference in the performance of the two models. This here uses a Monte
	# Carlo approximation of a paired permutation significance test.
	def computePermutationSignificance(self, truth, predictions1, predictions2):
		statistic1 = self.computeF1(truth, predictions1)
		statistic2 = self.computeF1(truth, predictions2)
		
		swap = 1 if (statistic1 > statistic2) else -1
		difference = (statistic1 - statistic2) * swap
		
		m = int(constants.PERMUTATIONS * 0.1)
		
		n = 0
		for i in range(constants.PERMUTATIONS):
			perm1, perm2 = self.permuteLabels(predictions1, predictions2)
			statistic1 = self.computeAccuracy(truth, perm1)
			statistic2 = self.computeAccuracy(truth, perm2)
			diff = (statistic1 - statistic2) * swap
	
			if diff >= difference:
				n += 1
	
			if i % m == 0:
				print "Permutation test, {0}% done.".format(int(i * 100 / constants.PERMUTATIONS))
	
		return (n + 1) / (constants.PERMUTATIONS + 1)


	# Shuffles paired labels with a probability of a fair coin toss.
	def permuteLabels(self, labels1, labels2):
		nLabels1 = []
		nLabels2 = []

		for i in range(len(labels1)):
			choice = [labels1[i], labels2[i]]
			random.shuffle(choice)
			nLabels1.append(choice[0])
			nLabels2.append(choice[1])

		return nLabels1, nLabels2
	
	
	
	### Predicted Language Similarity ###
	# Uses the learner to generate cognatenes predictions for every possible
	# word pair for every meaning. Uses these predictions to compute predicted
	# language pair similarity as a ratio of positive predictions to all
	# predictions.
	def predictLanguageSimilarity(self, model, wordforms, extractor):
		predictedCounts = self.countPredictions(model, wordforms, extractor)
		self.computeSimilarity(predictedCounts)
		

	# Generates a cognateness decision for each wordform and meaning, counts the
	# number of positive and all predictions for each language pair.
	def countPredictions(self, model, wordforms, extractor):
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
					
					if model == constants.SVM:
						prediction = self.predictSVM(extractor(form1, form2))
					elif model == constants.LR:
						prediction = self.predictLinearRegression(extractor(form1, form2))
					
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