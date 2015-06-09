from __future__ import division

import constants



### Printing to Terminal ###
# Prints to terminal key pairwise deduction results.
def reportPairwiseDeduction(prr, accuracy, report):
	print "\n", "### Pairwise Deduction ###"
	print constants.REPORTING.format("Negative test examples:", (1 - prr.positiveCounts[constants.TEST] / prr.allCounts[constants.TEST]))
	print "\n", constants.REPORTING.format("Accuracy:", accuracy)
	print "\n", report


# Prints to terminal key pairwise learning results.
def reportPairwiseLearning(stage, prr, accuracy, report):
	print "\n", "### Pairwise Learning (" + stage + ") ###"
	print constants.REPORTING.format("Negative examples:", (1 - prr.positiveCounts[constants.ALL] / prr.allCounts[constants.ALL]))
	print constants.REPORTING.format("Negative training examples:", (1 - prr.positiveCounts[constants.TRAIN] / prr.allCounts[constants.TRAIN]))
	print constants.REPORTING.format("Negative test examples:", (1 - prr.positiveCounts[constants.TEST] / prr.allCounts[constants.TEST]))
	print "\n", constants.REPORTING.format("Accuracy:", accuracy)
	print "\n", report


# Prints to terminal key group-based deduction results.
def reportGroup(scores, testMeanings, allMeanings):
	print "\n", "### Group-based Deduction ###"
	for i, meaningIndex in enumerate(testMeanings):
		print "{0:3d} {1:26} {2:.4f}".format(meaningIndex, allMeanings[meaningIndex], scores[i])
	print "\n", "{0:30} {1:.4f}".format("Average:", sum(scores) / len(scores)), "\n"


# Prints to terminal key clustering results.
def reportCluster(scores, distances, testMeanings, allMeanings):
	print "\n", "### Clustering ###"
	for i, meaningIndex in enumerate(testMeanings):
		print "{0:3d} {1:26} {2:.4f} {3:6.2f}".format(meaningIndex, allMeanings[meaningIndex], scores[i], distances[i])
	print "\n", "{0:30} {1:.4f} {2:6.2f}".format("Average:", sum(scores) / len(scores), sum(distances) / len(distances)), "\n"


### Saving to File ###
# Saves each example (a pair of wordforms with their languages) to a file
# together with their respective features and labels (both true and predicted).
def savePredictions(filename, examples, features, predictions, truth):
	with open(filename, "wb") as output:
		for i, (form1, form2, language1, language2) in enumerate(examples):
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