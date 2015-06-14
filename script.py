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
	
	# Output
	filename = "output/Pairwise " + constants.DEDUCERS[measure] + ".txt"
	output.savePredictions(filename, prr.examples[constants.TEST], ext.testExamples, predictedLabels, ext.testLabels)

	return predictedLabels, ext.testLabels


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
	V1scores = [lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings]
	
	# Reporting
	output.reportGroup(constants.DEDUCERS[measure], V1scores, prr.testMeanings, rdr.meanings)
	output.saveGroup("output/Group " + constants.DEDUCERS[measure] + ".txt", predictedSets)


def firstPassLearning(model, measure):
	# Feature extraction
	ext = extractor.Extractor()
	
	if measure == constants.MED:
		ext.MEDBaseline(prr.examples, prr.labels)
	elif measure == constants.HK2011:
		ext.HK2011Baseline(prr.examples, prr.labels)

	# Learning
	lrn = learner.Learner()
	stage = "1st Pass (" + constants.MODELS[model] + ", " + constants.MEASURES[measure] + ")"

	if model == constants.SVM:
		lrn.fitSVM(ext.trainExamples[:, : ext.testExamples.shape[1]], ext.trainLabels)
	elif model == constants.LR:
		lrn.fitLinearRegression(ext.trainExamples[:, : ext.testExamples.shape[1]], ext.trainLabels)

	# Prediction, evaluation, reporting and output
	predictedLabels, trueLabels = learningPipeline(model, ext, lrn, stage)
	
	return ext, lrn, predictedLabels, trueLabels


def secondPassLearning(ext, lrn, model, measure):
	# Feature extraction
	if measure == constants.MED:
		lrn.predictLanguageSimilarity(model, rdr.wordforms, ext.MEDExtractor)
	elif measure == constants.HK2011:
		lrn.predictLanguageSimilarity(model, rdr.wordforms, ext.HK2011Extractor)
	
	ext.appendTestSimilarities(lrn.predictedSimilarities, prr.examples)
	
	# Learning
	stage = "2nd Pass (" + constants.MODELS[model] + ", " + constants.MEASURES[measure] + ")"

	if model == constants.SVM:
		lrn.fitSVM(ext.trainExamples, ext.trainLabels)
	elif model == constants.LR:
		lrn.fitLinearRegression(ext.trainExamples, ext.trainLabels)
	
	# Prediction, evaluation, reporting and output
	predictedLabels, trueLabels = learningPipeline(model, ext, lrn, stage)
	
	return ext, lrn, predictedLabels, trueLabels


def clustering(ext, lrn, model, measure):
	# Feature extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)

	# Threshold
#	if measure == constants.MED:
#		threshold = lrn.computeDistanceThreshold(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.MEDExtractor, trueLabels)
#	elif measure == constants.HK2011:
#		threshold = lrn.computeDistanceThreshold(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor, trueLabels)
#	print "Threshold:", threshold

	# Learning
	if measure == constants.MED:
		predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(model, rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.MEDExtractor)
	elif measure == constants.HK2011:
		predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(model, rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor)
	
	# Evaluation
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}
	
	# Reporting
	output.reportCluster(V1scores, clusterCounts, clusterDistances, rdr.meanings)
	output.saveGroup("output/Clustering.txt", predictedSets)


def learningPipeline(model, ext, lrn, stage):
	# Predicting
	if model == constants.SVM:
		predictedLabels = lrn.predictSVM(ext.testExamples)
	elif model == constants.LR:
		predictedLabels = lrn.predictLinearRegression(ext.testExamples)
	
	# Evaluation
	accuracy = lrn.computeAccuracy(ext.testLabels, predictedLabels)
	report = lrn.evaluatePairwise(ext.testLabels, predictedLabels)
	
	# Reporting
	output.reportPairwiseLearning(stage, prr, accuracy, report)

	# Output
	output.savePredictions("output/" + stage + ".txt", prr.examples[constants.TEST], ext.testExamples, predictedLabels, ext.testLabels)

	return predictedLabels, ext.testLabels



# FLOW
# Reading
rdr = reader.Reader(constants.IN)
rdr.read()


# Pairing
prr = pairer.Pairer()
prr.pairByMeaningRatio(rdr.cognateCCNs, rdr.dCognateCCNs, constants.TRAIN_MEANINGS / constants.MEANING_COUNT)


# Pairwise Deduction
PDLabels1, trueLabels = pairwiseDeduction(constants.IDENTICAL_WORDS)
PDLabels2, _ = pairwiseDeduction(constants.IDENTICAL_PREFIX)


# Group Deduction
groupDeduction(constants.IDENTICAL_WORDS)


# Learning
ext, lrn, FPLabels1, _ = firstPassLearning(constants.SVM, constants.MED)
ext, lrn, SPLabels1, _ = secondPassLearning(ext, lrn, constants.SVM, constants.MED)
clustering(ext, lrn, constants.SVM, constants.MED)


# Significance
lrn = learner.Learner()
print "\nSignificance: {0} vs. {1}".format(constants.DEDUCERS[constants.IDENTICAL_WORDS], constants.DEDUCERS[constants.IDENTICAL_PREFIX])
print constants.SIGNIFICANCE.format(lrn.computeMcNemarSignificance(trueLabels, PDLabels1, PDLabels2))


# Output
#output.saveGroup("output/true_groups.txt", rdr.cognateSets)