# Settings
LANGUAGE_COUNT = 95
MEANING_COUNT = 200

PERMUTATIONS = 10000

# HK2011 1st pass
T1 = 0.3594
# HK2011 2nd pass
T2 = 0.3308
# Minimal approach
T3 = 0.3674
# Combined approach
T4 = 0.3667


# Data
IN = "input/input.txt"
POS = "input/POS.txt"
DOLGO = "input/dolgo.txt"
CONS = "input/consonants.txt"


# Types
HEADER = "a"
SUBHEADER = "b"
RELATIONSHIP = "c"


# CCNs
# No meaning (CCN0).
CCN0 = 0

# Unique words not cognate to anyting else (CCN1).
CCN1 = 1

# Cognate with each other, not cognate with other lists (CCN2).
CCN2_START = 2
CCN2_END = 99

# Doubtfully cognate with each other, not cognate with other lists (CCN3).
CCN3_START = 100
CCN3_END = 199

# Cognate with each other and cognate or doubtfully cognate with at least one
# form from some other list (CCN4).
CCN4_START = 200
CCN4_END = 399

# Doubtfully cognate with each other and doubtfully cognate with at least one
# form from some other list (CCN5).
CCN5_START = 400
CCN5_END = 499


# Relationships
COGNATION = 2
DOUBTFUL_COGNATION = 3


# Datasets
TRAIN = 0
TEST = 1


# Methods
IDENTICAL_WORDS = 0
IDENTICAL_PREFIX = 1
IDENTICAL_LETTER = 2

MED = 0
HK2011 = 1

SVM = 0
LR = 1

DEDUCERS = ["Identical Words", "Identical Prefixes", "Identical First Letter"]
MEASURES = ["Minimum Edit Distance", "Hauer & Kondrak, 2011"]
MODELS = ["SVM", "Logistic Regression"]


# Edit operations
EQUAL = "equal"
INSERT = "insert"
DELETE = "delete"
REPLACE = "replace"


# Classes
TARGETS = ["Non-cognates", "Cognates"]


# Letters
FIRST = ord("a")
LAST = ord("z")


# Languages
LANGUAGE_GROUPS = [
	# Celtic subfamily (7)
	range(1, 8),
	# Romance subfamily (16)
	range(8, 24),
	# Germanic subfamily (15)
	range(24, 39),
	# Baltoslavic subfamily (26)
	range(39, 55) + range(85, 95),
	# Indoaryan cluster (11)
	range(55, 66),
	# Greek subfamily (5)
	range(66, 71),
	# Armenian subfamily (2)
	range(71, 73),
	# Iranian cluster (7)
	range(73, 80),
	# Albanian subfamily (6)
	range(80, 85) + [95]
]


# Formatting
PICKLE_EXT = "pickles/ext{0}.pickle"
PICKLE_LRN = "pickles/lrn{0}.pickle"
REPORTING = "{0:30} {1:.4f}"
SIGNIFICANCE = "significance = {0:.5f}\n"