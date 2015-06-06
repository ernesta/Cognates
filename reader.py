from collections import OrderedDict
import re

import constants



class Reader:
	# Initializes the reader by setting the target filename and various data
	# structures that will be used for reading the file.
	def __init__(self, filename):
		# Input file name.
		self.filename = filename
		
		# Meanings and their indices (1 - 200).
		self.meanings = OrderedDict()
		
		# Languages and their indices (1 - 95).
		self.languages = {}
		
		# For each meaning, for each CCN, all other CCNs it is doubtfully
		# cognate with. This is used later when pairing cognates into positive
		# and negative examples to ensure that forms which are doubtfully
		# cognate are not used as examples of non-cognates.
		self.dCognateCCNs = {}
		
		# cognateCCNs is a dictionary whose keys are meaning indices. Each meaning
		# has an associated dictionary of cognate groups (CCNs). Each cognate
		# dictionary contains various CCNs and associated cognate dictionaries.
		# Each cognate dictionary is structured has language indices as keys and
		# language word forms for each given meaning as values. Due to issues in
		# the data, the input data is cleaned by removing entries that are
		# ambiguous or are have CCNs that indicate inconclusive cognateness
		# decisions.
		self.cognateCCNs = {}
		
		# For each meaning, the wordform for each language.
		self.wordforms = {}
		
		# For each meaning, for each cognate group, all wordforms that are
		# cognates of each other.
		self.cognateSets = {}
		
		# Meaning and language indices, CCN and cognate group that are currently
		# being processed.
		self.currentMeaningIndex = 0
		self.currentLanguageIndex = 0
		self.currentCCN = 0
		self.lastCCN = 0
		self.lastCognateGroup = -1


	# Reads the input file line by line, parses each line and populates the
	# associated data structures accordingly.
	def read(self):
		with open(self.filename, "rb") as data:
			for line in data:
				# Header line, indicates the beginning of a block.
				if line[0] == constants.HEADER:
					# Checks if the required amount of meanings has already been
					# processed.
					if self.currentMeaningIndex + 1 > constants.MEANING_COUNT:
						break
					
					self.processHeader(line)
		
				# Subheader line, indicates the beginning of a subblock.
				elif line[0] == constants.SUBHEADER:
					self.processSubheader(line)
			
				# Relationship line, describes relationships between two
				# subblocks.
				elif line[0] == constants.RELATIONSHIP:
					self.processRelationship(line)
	
				# Form line.
				else:
					# CCN0 forms are removed.
					if self.currentCCN == constants.CCN0:
						continue
					else:
						self.processForm(line)
	
		# Orders languages by their indices for better readability.
		self.languages = OrderedDict(sorted(self.languages.items(), key = lambda x: x[0]))


	# Processes the header line.
	def processHeader(self, line):
		splitLine = line.split()
			
		meaningIndex = int(splitLine[1])
		meaning = splitLine[2].lower()

		self.meanings[meaningIndex] = meaning
		self.cognateCCNs[meaningIndex] = {}
		
		self.currentMeaningIndex = meaningIndex


	# Processes the subheader line.
	def processSubheader(self, line):
		self.currentCCN = int(line.split()[1])
	
	
	# Processes the relationship line.
	def processRelationship(self, line):
		splitLine = line.split()
		
		firstCCN = int(splitLine[1])
		type = int(splitLine[2])
		secondCCN = int(splitLine[3])
		
		if type == constants.DOUBTFUL_COGNATION:
			self.addDoubtfulCCNs(firstCCN, secondCCN)
			self.addDoubtfulCCNs(secondCCN, firstCCN)


	# Processes the form line.
	def processForm(self, line):
		meaningIndex = int(line[2 : 5])
		languageIndex = int(line[6 : 8])
		language = line[9 : 24].strip().lower().title()

		form = self.parseForms(line)
		
		# Adds the form to the cognateCCNs dictionary. Also adds the form to its
		# appropriate cognate group based on its CCN.
		if form:
			if self.currentCCN not in self.cognateCCNs[meaningIndex]:
				self.cognateCCNs[meaningIndex][self.currentCCN] = {}
			self.cognateCCNs[meaningIndex][self.currentCCN][languageIndex] = form

			if meaningIndex not in self.wordforms:
				self.wordforms[meaningIndex] = {}
			self.wordforms[meaningIndex][languageIndex] = form
	
			self.addToCognateGroup(form)
	
		# If an unseen language is encountered, it is added to the language
		# dictionary.
		if languageIndex not in self.languages:
			self.languages[languageIndex] = language

		self.currentLanguageIndex = languageIndex


	# Adds a pair of CCNs to the doubtful CCN dictionary.
	def addDoubtfulCCNs(self, firstCCN, secondCCN):
		if self.currentMeaningIndex not in self.dCognateCCNs:
			self.dCognateCCNs[self.currentMeaningIndex] = {}
		
		if firstCCN not in self.dCognateCCNs[self.currentMeaningIndex]:
			self.dCognateCCNs[self.currentMeaningIndex][firstCCN] = []
	
		if secondCCN not in self.dCognateCCNs[self.currentMeaningIndex][firstCCN]:
			self.dCognateCCNs[self.currentMeaningIndex][firstCCN].append(secondCCN)
	
	
	# Parses a given form line to extract all wordforms.
	def parseForms(self, line):
		forms = []
		
		# While most multiple forms are provided using a comma-delimited list,
		# some are also delimited with a /.
		for form in re.split("/|,", line[25 :]):
			form = form.strip().lower()
			
			# Drops words that beging with - (these are not words but rather
			# indications of an alternative word ending).
			if form and form[0] == "-":
				form = ""
			
			# Drops parentheses, checks if after that the string is not empty.
			form = re.sub(r"\([^)]*\)", "", form).strip()
			
			if len(form) > 0:
				forms.append(form)
	
		# If there is more than one form, it becomes impossible to distinguish
		# between cognates and non-cognates for the particular language and
		# meaning. Thus, all entries with multiple forms are ignored (out of
		# 19,000 entries, 16,217 only contain one form).
		return forms[0] if len(forms) == 1 else None


	# Selects the approapriate cognate group to add the current wordform to.
	def addToCognateGroup(self, form):
		if self.currentMeaningIndex not in self.cognateSets:
			self.cognateSets[self.currentMeaningIndex] = {}
		
		# Each word in CCN1 gets its own cognate group.
		if self.currentCCN == constants.CCN1:
			self.lastCognateGroup += 1
			self.cognateSets[self.currentMeaningIndex][self.lastCognateGroup] = [(form, self.currentLanguageIndex)]

		# Each word in the same CCN2 goes to the same cognate group.
		# Each word in the same CCN4 goes to the same cognate group.
		elif ((self.currentCCN >= constants.CCN2_START) and (self.currentCCN <= constants.CCN2_END)) or ((self.currentCCN >= constants.CCN4_START) and (self.currentCCN <= constants.CCN4_END)):
			if self.currentCCN == self.lastCCN:
				self.cognateSets[self.currentMeaningIndex][self.lastCognateGroup].append((form, self.currentLanguageIndex))
			else:
				self.lastCognateGroup += 1
				self.cognateSets[self.currentMeaningIndex][self.lastCognateGroup] = [(form, self.currentLanguageIndex)]

		self.lastCCN = self.currentCCN