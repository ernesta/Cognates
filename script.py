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
	output.saveGroup("output/identical_letter_groups.txt", predictedSets)


def firstPassLearning():
	# Feature extraction
	ext = extractor.Extractor()
#	ext.MEDBaseline(prr.examples, prr.labels)
	ext.HK2011Baseline(prr.examples, prr.labels)

	# Learning
	lrn = learner.Learner()
	lrn.fitSVM(ext.trainExamples[:, : ext.testExamples.shape[1]], ext.trainLabels)

	# Prediction, evaluation, reporting and output
	prediction = learningPipeline(ext, lrn, "1st Pass", "output/HK2011First.txt")
	
	return ext, lrn


def secondPassLearning(ext, lrn):
	# Feature extraction
	lrn.predictLanguageSimilarity(rdr.wordforms, ext.HK2011Extractor)
	ext.appendTestSimilarities(lrn.predictedSimilarities, prr.examples)
	
	# Learning
	lrn.fitSVM(ext.trainExamples, ext.trainLabels)
	
	# Prediction, evaluation, reporting and output
	prediction = learningPipeline(ext, lrn, "2nd Pass", "output/HK2011Second.txt")
	
	return ext, lrn


def clustering(ext, lrn):
	# Feature Extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)

	# Threshold
#	threshold = lrn.computeDistanceThreshold(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor, trueLabels)
#	print "Threshold:", threshold

	# Learning
	predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor)
	
	# Evaluation
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}
	
	# Reporting
	output.reportCluster(V1scores, clusterCounts, clusterDistances, rdr.meanings)
	output.saveGroup("output/clustering.txt", predictedSets)


def learningPipeline(ext, lrn, stage, filename):
	# Predicting
	predictions = lrn.predictSVM(ext.testExamples)
	
	# Evaluation
	accuracy = lrn.computeAccuracy(ext.testLabels, predictions)
	report = lrn.evaluatePairwise(ext.testLabels, predictions)
	
	# Reporting
	output.reportPairwiseLearning(stage, prr, accuracy, report)

	# Output
	output.savePredictions(filename, prr.examples[constants.TEST], ext.testExamples, predictions, ext.testLabels)



# FLOW
# Reading
rdr = reader.Reader(constants.IN)
rdr.read()


# Pairing
prr = pairer.Pairer()
prr.pairByMeaningRatio(rdr.cognateCCNs, rdr.dCognateCCNs, 0.9)


# Deduction
#pairwiseDeduction()
#groupDeduction()

# First Pass
ext, lrn = firstPassLearning()
#output.pickleLearning(1, ext, lrn)


# Second Pass
#ext, lrn = output.unpickleLearning(1, extractor.Extractor(), learner.Learner())
ext, lrn = secondPassLearning(ext, lrn)


# Clustering
clustering(ext, lrn)


# Output
#output.saveGroup("output/true_groups.txt", rdr.cognateSets)