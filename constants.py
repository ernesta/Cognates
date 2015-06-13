# Settings
LANGUAGE_COUNT = 95
MEANING_COUNT = 200

CLUSTER_MIN = int(LANGUAGE_COUNT * 0.1)
CLUSTER_MAX = int(LANGUAGE_COUNT * 0.5)

PERMUTATIONS = 1000

THRESHOLD = 0.85


# Data
IN = "input.txt"


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
TEST = 2
ALL = 3


# METHODS
IDENTICAL_WORDS = 0
IDENTICAL_PREFIX = 1
IDENTICAL_LETTER = 2

MED = 0
HK2011 = 1

DEDUCERS = ["Identical Words", "Identical Prefixes", "Identical First Letter"]
LEARNERS = ["Minimum Edit Distance", "Hauer & Kondrak, 2011"]


# Classes
TARGETS = ["Non-cognates", "Cognates"]


# Formatting
REPORTING = "{0:30} {1:.4f}"
PICKLE_EXT = "pickles/ext{0}.pickle"
PICKLE_LRN = "pickles/lrn{0}.pickle"