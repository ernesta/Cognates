from __future__ import division
import pickle

import constants



### Printing to Terminal ###
# Prints to terminal key pairwise deduction results.
def reportPairwiseDeduction(stage, prr, accuracy, F1, report):
	print "\n", "### Pairwise Deduction (" + stage + ") ###"
	print constants.REPORTING.format("Negative test examples:", prr.negativeCounts[constants.TEST] / (prr.negativeCounts[constants.TEST] + prr.positiveCounts[constants.TEST]))
	print "\n", constants.REPORTING.format("Accuracy:", accuracy)
	print constants.REPORTING.format("F1:", F1)
	print "\n", report


# Prints to terminal key pairwise learning results.
def reportPairwiseLearning(stage, prr, accuracy, F1, report):
	print "\n", "### Pairwise Learning (" + stage + ") ###"
	print constants.REPORTING.format("Negative examples:", (prr.negativeCounts[constants.TRAIN] + prr.negativeCounts[constants.TEST]) / (prr.negativeCounts[constants.TRAIN] + prr.negativeCounts[constants.TEST] + prr.positiveCounts[constants.TRAIN] + prr.positiveCounts[constants.TEST]))
	print constants.REPORTING.format("Negative training examples:", prr.negativeCounts[constants.TRAIN] / (prr.negativeCounts[constants.TRAIN] + prr.positiveCounts[constants.TRAIN]))
	print constants.REPORTING.format("Negative test examples:", prr.negativeCounts[constants.TEST] / (prr.negativeCounts[constants.TEST] + prr.positiveCounts[constants.TEST]))
	print "\n", constants.REPORTING.format("Accuracy:", accuracy)
	print constants.REPORTING.format("F1:", F1)
	print "\n", report


# Prints to terminal key group-based deduction results.
def reportGroup(stage, scores, allMeanings):
	print "\n", "### Group-based Deduction (" + stage + ") ###"
	for meaningIndex in sorted(scores.keys()):
		print "{0:3d} {1:26} {2:.4f}".format(meaningIndex, allMeanings[meaningIndex], scores[meaningIndex])
	print "\n", "{0:30} {1:.4f}".format("Average:", sum(scores.values()) / len(scores)), "\n"


# Prints to terminal key clustering results.
def reportCluster(scores, counts, distances, allMeanings):
	print "\n", "### Clustering ###"
	for meaningIndex in sorted(scores.keys()):
		print "{0:3d} {1:26} {2:2d} {3:.4f} {4:6.2f}".format(meaningIndex, allMeanings[meaningIndex], counts[meaningIndex], scores[meaningIndex], distances[meaningIndex])
	print "\n", "{0:30} {1:2d} {2:.4f} {3:6.2f}".format("Average:", int(sum(counts.values()) / len(counts)), sum(scores.values()) / len(scores), sum(distances.values()) / len(distances)), "\n"


### Saving to File ###
# Saves each example (a pair of wordforms with their languages) to a file
# together with their respective features and labels (both true and predicted).
def savePredictions(filename, examples, features, predictions, truth):
	with open(filename, "wb") as output:
		for i, (form1, form2, language1, language2, meaningIndex) in enumerate(examples):
			sExample = "{0} ({1}), {2} ({3})".format(form1, language1, form2, language2)
			sFeatures = "[" + "  ".join(["{0:4.1f}".format(feature) for feature in features[i]]) + " ]"
			
			output.write("{0:40} {1:20} {2:2} {3:2}\n".format(sExample, sFeatures, truth[i], int(predictions[i])))


# Saves a readable version of clustering of grouping to a file.
def saveGroup(filename, predictedSets):
	with open(filename, "wb") as output:
		for meaningIndex, groups in predictedSets.iteritems():
			output.write("Meaning: {0}\n".format(meaningIndex))

			for groupIndex, entries in groups.iteritems():
				output.write("Group: {0}\n".format(groupIndex))
				
				for (wordform, languageIndex) in entries:
					output.write("{0} ({1})\n".format(wordform, languageIndex))


### Serialization ###
# Pickles data stored in extractor and learner objects.
def pickleLearning(stage, ext, lrn):
	extData = [ext.trainExamples, ext.trainLabels, ext.testExamples, ext.testLabels]
	lrnData = [lrn.scaler, lrn.machine, lrn.predictedSimilarities]
	
	with open(constants.PICKLE_EXT.format(stage), "wb") as output:
		pickle.dump(extData, output)

	with open(constants.PICKLE_LRN.format(stage), "wb") as output:
		pickle.dump(lrnData, output)


# Unpickles pickled data, adds it to provided extractor and learner objects.
def unpickleLearning(stage, ext, lrn):
	with open(constants.PICKLE_EXT.format(stage), "rb") as input:
		extData = pickle.load(input)

	with open(constants.PICKLE_LRN.format(stage), "rb") as input:
		lrnData = pickle.load(input)

	ext.trainExamples = extData[0]
	ext.trainLabels = extData[1]
	ext.testExamples = extData[2]
	ext.testLabels = extData[3]

	lrn.scaler = lrnData[0]
	lrn.machine = lrnData[1]
	lrn.predictedSimilarities = lrnData[2]

	return ext, lrn