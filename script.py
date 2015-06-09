import numpy

import constants
import extractor
import learner
import output
import pairer
import reader



# FUNCTIONS
def pairwiseDeduction():
	# Feature extraction
	ext = extractor.Extractor()
	ext.identicalWordsBaseline(prr.examples, prr.labels)

	# Evaluation
	lrn = learner.Learner()
	accuracy = lrn.computeAccuracy(ext.testLabels, ext.testExamples)
	report = lrn.evaluatePairwise(ext.testLabels, ext.testExamples)

	# Reporting
	output.reportPairwiseDeduction(prr, accuracy, report)
	
	# Output
	filename = "output/identical_word.txt"
	output.savePredictions(filename, prr.examples[constants.TEST], ext.testExamples, numpy.array(ext.testExamples), ext.testLabels)


def groupDeduction():
	# Feature extraction
	ext = extractor.Extractor()
	predictedLabels, predictedSets = ext.identicalFirstLettersGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	
	# Evaluation
	lrn = learner.Learner()
	V1scores = [lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings]
	
	# Reporting
	output.reportGroup(V1scores, prr.testMeanings, rdr.meanings)


def firstPassLearning():
	# Feature extraction
	ext = extractor.Extractor()
	ext.HK2011Baseline(prr.examples, prr.labels)
	
	# Learning
	lrn = learner.Learner()
	lrn.fitSVM(ext.trainExamples[:, : ext.trainExamples.shape[1] - 1], ext.trainLabels)
	
	# Prediction, evaluation, reporting and output
	prediction = learningPipeline(ext, lrn, "output/HK2011First.txt")
	
	return ext, lrn


def secondPassLearning(ext, lrn):
	# Feature extraction
	lrn.predictLanguageSimilarity(rdr.wordforms, ext.HK2011Extractor)
	ext.appendTestSimilarities(lrn.predictedSimilarities, prr.examples)
	
	# Learning
	lrn.fitSVM(ext.trainExamples, ext.trainLabels)
	
	# Prediction, evaluation, reporting and output
	prediction = learningPipeline(ext, lrn, "output/HK2011Second.txt")
	
	return ext, lrn


def clustering(ext, lrn):
	# Feature Extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	
	# Learning
	predictedLabels, predictedSets, minDistances = lrn.cluster(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor)
	
	# Evaluation
	V1scores = [lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings]
	
	# Reporting
	output.reportCluster(V1scores, minDistances, prr.testMeanings, rdr.meanings)


def learningPipeline(ext, lrn, filename):
	# Predicting
	predictions = lrn.predictSVM(ext.testExamples)
	
	# Evaluation
	accuracy = lrn.computeAccuracy(ext.testLabels, predictions)
	report = lrn.evaluatePairwise(ext.testLabels, predictions)
	
	# Reporting
	output.reportPairwiseLearning(prr, accuracy, report)

	# Output
	output.savePredictions(filename, prr.examples[constants.TEST], ext.testExamples, predictions, ext.testLabels)



# FLOW
# Reading
rdr = reader.Reader(constants.IN)
rdr.read()

# Pairing
prr = pairer.Pairer()
prr.pairByLanguageRatio(rdr.cognateCCNs, rdr.dCognateCCNs, 0.5)
#prr.pairByMeaningRatio(rdr.cognateCCNs, rdr.dCognateCCNs, 0.5)

# Deduction
pairwiseDeduction()
groupDeduction()

# Learning
ext, lrn = firstPassLearning()
ext, lrn = secondPassLearning(ext, lrn)

# Clustering
clustering(ext, lrn)