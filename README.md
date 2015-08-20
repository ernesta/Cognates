# Cognates

This is the code base for a MSc Artificial Intelligence dissertation at the University of Edinburgh. The best readme for the code is the [dissertation](http://ssrn.com/abstract=2647558) itself, so I'll be concise.

Cognates are words in different languages that have evolved from the same proto-form. They often look and sound similar (e.g., Danish *nat* and Lithuanian *naktis*, both meaning *night*). Sometimes, however, they seem to be completely unrelated (e.g., Faroese *hjarta* and Lithuanian *širdis* for *heart*). Historical linguists use the comparative method (a bunch of heuristics, really) to figure out if a pair of words is related genetically. The method is very precise, but also extremely time-consuming. With around 3,000 languages (out of 7,000+ currently spoken in the world) expected to be dead by the end of the century, this is not good enough. This project is an attempt to automate cognate identification, focusing specifically on little-known under-resourced languages.


## Data
Data is stored in the input directory:
+ *input.txt:* the raw Comparative Indo-European Database (Dyen, Kruskal, & Black, 1992). The only difference from [the original](http://www.wordgumbo.com/ie/cmp/) is the removal of the introduction to the file.
+ *clean.txt:* a cleaned-up version of the same file. Not used in the project, but a similar representation is stored in memory whenever the raw file is read. Good for use in other projects or manual browsing.
+ *POS.txt:* POS tags for each of the 200 meanings from the Comparative Indo-European Database. Assigned manually by the author.
+ *consonants.txt:* a list of Roman characters considered to be consonants.
+ *dolgo.txt:* Dolgopolsky's (1986) sound classes and their corresponding Roman characters.

## Code
+ *script.py:* controls the flow of the program.
+ *constants.py:* exactly that.
+ *reader.py:* reads the Comparative Indo-European Database, performs data cleaning.
+ *pairer.py:* pairs words within each meaning, creating positive and negative examples for classification. Divides the paired data into training, development, and test sets.
+ *extractor.py:* given a pair of words, extracts various features (string similarity, letter correspondences, POS tags, and language groups).
+ *learner.py:* implements SVM and logistic regression classifiers, hierarchical agglomerative clustering, and a number of evaluation metrics.

## Libraries

+ [scikit-learn](http://scikit-learn.org/stable/)
+ [NumPy](http://www.numpy.org/)
+ [python-Levenshtein](https://pypi.python.org/pypi/python-Levenshtein/)

## Author
**Ernesta Orlovaitė**

+ [ernes7a.lt](http://ernes7a.lt)
+ [@ernes7a](http://twitter.com/ernes7a)