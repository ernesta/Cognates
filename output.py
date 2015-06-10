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
def reportCluster(scores, counts, distances, allMeanings):
	print "\n", "### Clustering ###"
	for meaningIndex, score in scores.iteritems():
		print "{0:3d} {1:26} {2:2d} {3:.4f} {4:6.2f}".format(meaningIndex, allMeanings[meaningIndex], counts[meaningIndex], score, distances[meaningIndex])
	print "\n", "{0:30} {1:2d} {2:.4f} {3:6.2f}".format("Average:", int(sum(counts.values()) / len(counts)), sum(scores.values()) / len(scores), sum(distances.values()) / len(distances)), "\n"


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