from __future__ import division
import itertools
import operator

import constants
import extractor
import learner
import output
import pairer
import reader



# FUNCTIONS
### Rule-Based Baselines ###
def pairwiseDeduction(measure):
	# Feature extraction
	ext = extractor.Extractor()
	
	if measure == constants.IDENTICAL_WORDS:
		ext.identicalWordsBaseline(prr.examples, prr.labels)
	elif measure == constants.IDENTICAL_PREFIX:
		ext.identicalPrefixBaseline(prr.examples, prr.labels)
	elif measure == constants.IDENTICAL_LETTER:
		ext.identicalFirstLetterBaseline(prr.examples, prr.labels)
	
	predictions = ext.testExamples.reshape((ext.testExamples.shape[0],))

	# Evaluation
	lrn = learner.Learner()
	accuracy = lrn.computeAccuracy(ext.testLabels, predictions)
	F1 = lrn.computeF1(ext.testLabels, predictions)
	report = lrn.evaluatePairwise(ext.testLabels, predictions)
	
	# Reporting
	output.reportPairwiseDeduction(constants.DEDUCERS[measure], prr, accuracy, F1, report)
	output.savePredictions("output/Pairwise " + constants.DEDUCERS[measure] + ".txt", prr.examples[constants.TEST], ext.testExamples, predictions, ext.testLabels)

	return predictions


def groupDeduction(measure):
	# Feature extraction
	ext = extractor.Extractor()
	
	if measure == constants.IDENTICAL_WORDS:
		predictedLabels, predictedSets = ext.identicalWordsGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)
	elif measure == constants.IDENTICAL_PREFIX:
		predictedLabels, predictedSets = ext.identicalPrefixGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)
	elif measure == constants.IDENTICAL_LETTER:
		predictedLabels, predictedSets = ext.identicalFirstLetterGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)

	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	
	# Evaluation
	lrn = learner.Learner()
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}

	# Reporting
	output.reportGroup(constants.DEDUCERS[measure], V1scores, rdr.meanings)
	output.saveGroup("output/Group " + constants.DEDUCERS[measure] + ".txt", predictedSets)


### Hauer & Kondrak, 2011 ###
def HK2011Pairwise(twoStage = False):
	# 1st Pass
	# Feature extraction
	ext = extractor.Extractor()
	ext.HK2011Baseline(prr.examples, prr.labels)

	# Learning
	lrn = learner.Learner()
	lrn.initSVM(0.1)
	lrn.fitSVM(ext.trainExamples, ext.trainLabels)
	
	# Prediction
	predictions1 = lrn.predictSVM(ext.testExamples)
	
	# Evaluation
	accuracy = lrn.computeAccuracy(ext.testLabels, predictions1)
	F1 = lrn.computeF1(ext.testLabels, predictions1)
	report = lrn.evaluatePairwise(ext.testLabels, predictions1)
	
	# Reporting
	stage = "HK2011 1st Pass"
	output.reportPairwiseLearning(stage, prr, accuracy, F1, report)
	output.savePredictions("output/" + stage + ".txt", prr.examples[constants.TEST], ext.testExamples, predictions1, ext.testLabels)
	
	# 2nd Pass
	if twoStage:
		# Feature extraction
		ext.appendBinaryLanguageFeatures(prr.examples, prr.labels, constants.TEST, prr.testLanguages)

		# Learning
		lrn = learner.Learner()
		lrn.initSVM(0.001)
		lrn.fitSVM(ext.testExamples, predictions1)
	
		# Prediction
		predictions2 = lrn.predictSVM(ext.testExamples)
	
		# Evaluation
		accuracy = lrn.computeAccuracy(ext.testLabels, predictions2)
		F1 = lrn.computeF1(ext.testLabels, predictions2)
		report = lrn.evaluatePairwise(ext.testLabels, predictions2)
	
		# Reporting
		stage = "HK2011 2nd Pass"
		output.reportPairwiseLearning(stage, prr, accuracy, F1, report)
		output.savePredictions("output/" + stage + ".txt", prr.examples[constants.TEST], ext.testExamples, predictions2, ext.testLabels)

		# Significance
		print constants.SIGNIFICANCE.format(lrn.computeMcNemarSignificance(ext.testLabels, predictions1, predictions2))
	
	return ext, lrn


def HK2011Clustering(ext, lrn, twoStage = False):
	# Feature extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	extractor = ext.HK2011ExtractorFull if twoStage else ext.HK2011Extractor

	# Learning
	threshold = constants.T2 if twoStage else constants.T1
	predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(constants.SVM, threshold, rdr.wordforms, rdr.POSTags, prr.testMeanings, prr.testLanguages, extractor)

	# Evaluation
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}

	# Reporting
	output.reportCluster(V1scores, clusterCounts, clusterDistances, rdr.meanings)
	output.saveGroup("output/Clustering.txt", predictedSets)


### Combined Approach ###
def treeFeatureSelection():
	# Feature extraction
	ext = extractor.Extractor()
	ext.appendWordSimilarityFeatures(prr.examples, prr.labels, ext.allMeasures)
	
	# Feature selection
	lrn = learner.Learner()
	lrn.initForest(250, 0)
	lrn.fitForest(ext.trainExamples, ext.trainLabels)
	importances = lrn.getForestImportances()
	
	# Reporting
	for i, feature in enumerate(ext.allMeasures):
		print "{0}: {1:.4f}".format(feature, importances[i])


def editOperations():
	# Feature extraction
	ext = extractor.Extractor()
	operations = ext.extractEditOps(prr.examples, prr.labels)


def pairwiseLearning(minimal = False):
	# Feature extraction
	ext = extractor.Extractor()
	ext.consonantPrep = rdr.consonants
	ext.soundClassPrep = rdr.soundClasses
	
	if minimal:
		ext.appendWordSimilarityFeatures(prr.examples, prr.labels, ext.minimalMeasures)
		ext.appendPOSTags(prr.examples, prr.labels, rdr.POSTags)
	else:
		ext.appendWordSimilarityFeatures(prr.examples, prr.labels, [ext.commonBigramRatio, ext.commonTrigramNumber, ext.bigramDice, ext.jaroDistance])
		ext.appendWordSimilarityFeatures(prr.examples, prr.labels, [ext.identicalWords], rdr.consonants)
		ext.appendWordSimilarityFeatures(prr.examples, prr.labels, [ext.LCPLength, ext.commonBigramNumber, ext.identicalPrefix], rdr.soundClasses)
		ext.appendPOSTags(prr.examples, prr.labels, rdr.POSTags)
		ext.appendLetterFeatures(prr.examples, prr.labels)
		ext.appendSameLanguageGroupFeatures(prr.examples, prr.labels)

	# Learning
	lrn, predictions = learn(ext, 0.0001)

	# Reporting
	stage = "Pairwise Learning"
	accuracy = lrn.computeAccuracy(ext.testLabels, predictions)
	F1 = lrn.computeF1(ext.testLabels, predictions)
	report = lrn.evaluatePairwise(ext.testLabels, predictions)
	
	output.reportPairwiseLearning(stage, prr, accuracy, F1, report)
	output.savePredictions("output/" + stage + ".txt", prr.examples[constants.TEST], ext.testExamples, predictions, ext.testLabels)

	return ext, lrn


def groupLearning(ext, lrn, minimal = False):
	# Feature extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	extractor = ext.minimalExtractor if minimal else ext.combinedExtractor

	# Learning
	threshold = constants.T3 if minimal else constants.T4
	predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(constants.LR, threshold, rdr.wordforms, rdr.POSTags, prr.testMeanings, prr.testLanguages, extractor)
	
	# Evaluation
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}
	
	# Reporting
	output.reportCluster(V1scores, clusterCounts, clusterDistances, rdr.meanings)
	output.saveGroup("output/Clustering.txt", predictedSets)


def learn(ext, C):
	# Learning
	lrn = learner.Learner()
	lrn.initLogisticRegression(C)
	lrn.fitLogisticRegression(ext.trainExamples, ext.trainLabels)
	
	# Prediction
	predictions = lrn.predictLogisticRegression(ext.testExamples)
	
	return lrn, predictions



# FLOW
# Reading
rdr = reader.Reader()
rdr.read()

trainMeanings = [i for i in range(1, constants.MEANING_COUNT + 1) if (i % 10 != 0 and i % 10 != 5)]
devMeanings = [i for i in range(1, constants.MEANING_COUNT + 1) if i % 10 == 5]
testMeanings = [i for i in range(1, constants.MEANING_COUNT + 1) if i % 10 == 0]


# Pairing
prr = pairer.Pairer()
prr.pairBySpecificMeaning(rdr.cognateCCNs, rdr.dCognateCCNs, trainMeanings, testMeanings)


# Learning