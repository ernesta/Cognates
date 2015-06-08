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
	accuracy = lrn.computeSVMAccuracy(ext.testLabels, ext.testExamples)
	report = lrn.evaluateSVM(ext.testLabels, ext.testExamples)

	# Reporting
	output.reportDeduction(prr, accuracy, report)
	
	# Output
	filename = "output/identical_word.txt"
	output.savePredictions(filename, prr.examples[constants.TRAIN] + prr.examples[constants.TEST], ext.testExamples, numpy.array(ext.testExamples), ext.testLabels)


def groupDeduction():
	# Feature extraction
	ext = extractor.Extractor()


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
	lrn.predictLanguageSimilarity(rdr.wordforms, ext.HK2011Extractor)
	ext.appendTestSimilarities(lrn.predictedSimilarities, prr.examples)
	
	# Learning
	lrn.fitSVM(ext.trainExamples, ext.trainLabels)
	
	# Prediction, evaluation, reporting and output
	prediction = learningPipeline(ext, lrn, "output/HK2011Second.txt")
	
	return ext, lrn


def clustering(ext, lrn):
	predictedSets = lrn.cluster(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor)


def learningPipeline(ext, lrn, filename):
	# Predicting
	predictions = lrn.predictSVM(ext.testExamples)
	
	# Evaluation
	accuracy = lrn.computeSVMAccuracy(ext.testLabels, predictions)
	report = lrn.evaluateSVM(ext.testLabels, predictions)
	
	# Reporting
	output.reportLearning(prr, accuracy, report)

	# Output
	output.savePredictions(filename, prr.examples[constants.TEST], ext.testExamples, predictions, ext.testLabels)



# FLOW
# Reading
rdr = reader.Reader(constants.IN)
rdr.read()

# Pairing
prr = pairer.Pairer()
prr.pairByLanguageRatio(rdr.cognateCCNs, rdr.dCognateCCNs, len(rdr.languages), 0.5)
#prr.pairByMeaningRatio(rdr.cognateCCNs, rdr.dCognateCCNs, 0.5)

# Deduction
pairwiseDeduction()
groupDeduction()

# Learning
ext, lrn = firstPassLearning()
ext, lrn = secondPassLearning(ext, lrn)

# Clustering
clustering(ext, lrn)