from __future__ import division

import numpy

import constants
import extractor
import learner
import output
import pairer
import reader



# FUNCTIONS
def pairwiseDeduction(measure):
	# Feature extraction
	ext = extractor.Extractor()
	
	if measure == constants.IDENTICAL_WORDS:
		ext.identicalWordsBaseline(prr.examples, prr.labels)
	elif measure == constants.IDENTICAL_PREFIX:
		ext.identicalPrefixBaseline(prr.examples, prr.labels)
	elif measure == constants.IDENTICAL_LETTER:
		ext.identicalFirstLetterBaseline(prr.examples, prr.labels)
	
	predictedLabels = ext.testExamples.reshape((ext.testExamples.shape[0],))

	# Evaluation
	lrn = learner.Learner()
	accuracy = lrn.computeAccuracy(ext.testLabels, ext.testExamples)
	report = lrn.evaluatePairwise(ext.testLabels, ext.testExamples)
	
	# Reporting
	output.reportPairwiseDeduction(constants.DEDUCERS[measure], prr, accuracy, report)
	output.savePredictions("output/Pairwise " + constants.DEDUCERS[measure] + ".txt", prr.examples[constants.TEST], ext.testExamples, predictedLabels, ext.testLabels)

	return predictedLabels


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


def HK2011Pairwise():
	# 1st Pass
	# Feature extraction
	ext = extractor.Extractor()
	ext.HK2011Baseline(prr.examples, prr.labels)

#	ext.appendLanguageFeatures(prr.examples, constants.TRAIN, prr.trainLanguages)
#	ext.appendLanguageFeatures(prr.examples, constants.TEST, prr.testLanguages)

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
	# Feature extraction
	ext.appendLanguageFeatures(prr.examples, constants.TEST, prr.testLanguages)

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


#	# Significance
#	print constants.SIGNIFICANCE.format(lrn.computeMcNemarSignificance(ext.testLabels, predictions1, predictions2))


	return ext, lrn



def HK2011Clustering(ext, lrn):
	# Feature extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	
	# Threshold
#	threshold = lrn.computeDistanceThreshold(constants.SVM, rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor, trueLabels)
#	print "Threshold:", threshold

	# Learning
	predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(constants.SVM, constants.T1, rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011ExtractorFull)

	# Evaluation
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}

	# Reporting
	output.reportCluster(V1scores, clusterCounts, clusterDistances, rdr.meanings)
	output.saveGroup("output/Clustering.txt", predictedSets)



# FLOW
# Reading
rdr = reader.Reader(constants.IN)
rdr.read()

trainMeanings = [i for i in range(1, constants.MEANING_COUNT + 1) if (i % 10 != 0 and i % 10 != 5)]
devMeanings = [i for i in range(1, constants.MEANING_COUNT + 1) if i % 10 == 5]
testMeanings = [i for i in range(1, constants.MEANING_COUNT + 1) if i % 10 == 0]


# Pairing
prr = pairer.Pairer()
prr.pairBySpecificMeaning(rdr.cognateCCNs, rdr.dCognateCCNs, trainMeanings, testMeanings)


# Learning
ext, lrn = HK2011Pairwise()
HK2011Clustering(ext, lrn)