from __future__ import division

import constants



# Prints to terminal key deduction results.
def reportDeduction(prr, accuracy, report):
	print "\n", "### Pairwise Deduction ###"
	print constants.REPORTING.format("Negative examples:", (1 - prr.positiveCounts[constants.ALL] / prr.allCounts[constants.ALL]))
	print "\n", constants.REPORTING.format("Accuracy:", accuracy)
	print "\n", report


# Prints to terminal key learning results.
def reportLearning(prr, accuracy, report):
	print "\n", "### Pairwise Learning (1st Pass) ###"
	print constants.REPORTING.format("Negative examples:", (1 - prr.positiveCounts[constants.ALL] / prr.allCounts[constants.ALL]))
	print constants.REPORTING.format("Negative training examples:", (1 - prr.positiveCounts[constants.TRAIN] / prr.allCounts[constants.TRAIN]))
	print constants.REPORTING.format("Negative test examples:", (1 - prr.positiveCounts[constants.TEST] / prr.allCounts[constants.TEST]))
	print "\n", constants.REPORTING.format("Accuracy:", accuracy)
	print "\n", report



# Saves each example (a pair of wordforms with their languages) to a file
# together with their respective features and labels (both true and predicted).
def savePredictions(filename, examples, features, predictions, truth):
	with open(filename, "wb") as output:
		for i, (form1, form2, language1, language2) in enumerate(examples):
			sExample = "{0} ({1}), {2} ({3})".format(form1, language1, form2, language2)
			sFeatures = "[" + "  ".join(["{0:4.1f}".format(feature) for feature in features[i]]) + " ]"
			
			output.write("{0:40} {1:20} {2:2} {3:2}\n".format(sExample, sFeatures, truth[i], int(predictions[i])))


