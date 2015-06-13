import numpy

import constants
import extractor
import learner
import output
import pairer
import reader



# FUNCTIONS
def pairwiseDeduction(method):
	# Feature extraction
	ext = extractor.Extractor()
	
	if method == constants.IDENTICAL_WORDS:
		ext.identicalWordsBaseline(prr.examples, prr.labels)
	elif method == constants.IDENTICAL_PREFIX:
		ext.identicalPrefixBaseline(prr.examples, prr.labels)
	elif method == constants.IDENTICAL_LETTER:
		ext.identicalFirstLetterBaseline(prr.examples, prr.labels)
	
	predictedLabels = ext.testExamples.reshape((ext.testExamples.shape[0],))

	# Evaluation
	lrn = learner.Learner()
	accuracy = lrn.computeAccuracy(ext.testLabels, ext.testExamples)
	report = lrn.evaluatePairwise(ext.testLabels, ext.testExamples)
	
	# Reporting
	output.reportPairwiseDeduction(constants.DEDUCERS[method], prr, accuracy, report)
	
	# Output
	filename = "output/Pairwise " + constants.DEDUCERS[method] + ".txt"
	output.savePredictions(filename, prr.examples[constants.TEST], ext.testExamples, predictedLabels, ext.testLabels)

	return predictedLabels, ext.testLabels


def groupDeduction(method):
	# Feature extraction
	ext = extractor.Extractor()
	
	if method == constants.IDENTICAL_WORDS:
		predictedLabels, predictedSets = ext.identicalWordsGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)
	elif method == constants.IDENTICAL_PREFIX:
		predictedLabels, predictedSets = ext.identicalPrefixGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)
	elif method == constants.IDENTICAL_LETTER:
		predictedLabels, predictedSets = ext.identicalFirstLetterGroupBaseline(prr.testMeanings, prr.testLanguages, rdr.wordforms)

	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)
	
	# Evaluation
	lrn = learner.Learner()
	V1scores = [lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings]
	
	# Reporting
	output.reportGroup(constants.DEDUCERS[method], V1scores, prr.testMeanings, rdr.meanings)
	output.saveGroup("output/Group " + constants.DEDUCERS[method] + ".txt", predictedSets)


def firstPassLearning(method):
	# Feature extraction
	ext = extractor.Extractor()
	
	if method == constants.MED:
		ext.MEDBaseline(prr.examples, prr.labels)
	elif method == constants.HK2011:
		ext.HK2011Baseline(prr.examples, prr.labels)

	# Learning
	lrn = learner.Learner()
	lrn.fitSVM(ext.trainExamples[:, : ext.testExamples.shape[1]], ext.trainLabels)

	# Prediction, evaluation, reporting and output
	predictedLabels, trueLabels = learningPipeline(ext, lrn, "1st Pass (" + constants.LEARNERS[method] + ")")
	
	return ext, lrn, predictedLabels, trueLabels


def secondPassLearning(ext, lrn, method):
	# Feature extraction
	if method == constants.MED:
		lrn.predictLanguageSimilarity(rdr.wordforms, ext.MEDExtractor)
	elif method == constants.HK2011:
		lrn.predictLanguageSimilarity(rdr.wordforms, ext.HK2011Extractor)
	
	ext.appendTestSimilarities(lrn.predictedSimilarities, prr.examples)
	
	# Learning
	lrn.fitSVM(ext.trainExamples, ext.trainLabels)
	
	# Prediction, evaluation, reporting and output
	predictedLabels, trueLabels = learningPipeline(ext, lrn, "2nd Pass (" + constants.LEARNERS[method] + ")")
	
	return ext, lrn, predictedLabels, trueLabels


def clustering(ext, lrn, method):
	# Feature Extraction
	trueLabels = ext.extractGroupLabels(rdr.cognateSets, rdr.wordforms, prr.testMeanings, prr.testLanguages)

	# Threshold
#	if method == constants.MED:
#		threshold = lrn.computeDistanceThreshold(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.MEDExtractor, trueLabels)
#	elif method == constants.HK2011:
#		threshold = lrn.computeDistanceThreshold(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor, trueLabels)
#	print "Threshold:", threshold

	# Learning
	if method == constants.MED:
		predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.MEDExtractor)
	elif method == constants.HK2011:
		predictedLabels, predictedSets, clusterCounts, clusterDistances = lrn.cluster(rdr.wordforms, prr.testMeanings, prr.testLanguages, ext.HK2011Extractor)
	
	# Evaluation
	V1scores = {meaningIndex: lrn.computeV1(trueLabels[meaningIndex], predictedLabels[meaningIndex]) for meaningIndex in prr.testMeanings}
	
	# Reporting
	output.reportCluster(V1scores, clusterCounts, clusterDistances, rdr.meanings)
	output.saveGroup("output/Clustering.txt", predictedSets)


def learningPipeline(ext, lrn, stage):
	# Predicting
	predictedLabels = lrn.predictSVM(ext.testExamples)
	
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
prr.pairByMeaningRatio(rdr.cognateCCNs, rdr.dCognateCCNs, 0.9)


# Pairwise Deduction
PDLabels1, trueLabels = pairwiseDeduction(constants.IDENTICAL_WORDS)
PDLabels2, _ = pairwiseDeduction(constants.IDENTICAL_PREFIX)
PDLabels3, _ = pairwiseDeduction(constants.IDENTICAL_LETTER)


# Group Deduction
groupDeduction(constants.IDENTICAL_WORDS)
groupDeduction(constants.IDENTICAL_PREFIX)
groupDeduction(constants.IDENTICAL_LETTER)


# Learning
ext, lrn, FPLabels1, _ = firstPassLearning(constants.MED)
ext, lrn, SPLabels1, _ = secondPassLearning(ext, lrn, constants.MED)
clustering(ext, lrn, constants.MED)

ext, lrn, FPLabels2, _ = firstPassLearning(constants.HK2011)
ext, lrn, SPLabels2, _ = secondPassLearning(ext, lrn, constants.HK2011)
clustering(ext, lrn, constants.HK2011)

# Significance
print lrn.computePairwiseSignificance(trueLabels, PDLabels1, PDLabels2)


# Output
#output.saveGroup("output/true_groups.txt", rdr.cognateSets)