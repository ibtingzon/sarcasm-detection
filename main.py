import string
import nltk
import os
import sys  
import operator
import json
from nltk import bigrams
from nltk import trigrams
from nltk.corpus import stopwords

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from stemming.porter2 import stem
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from twython import Twython

cwd = os.getcwd()
reload(sys)  
sys.setdefaultencoding('utf8')
stop_words = set(stopwords.words('english'))
classes = ['sarc', 'notsarc']
model_dir = 'models'
persist = 'persist'

def test(classifiers, x_test):
	print x_test
	unigrams, bigrams, all_pos_tags = loadData()
	uni_tokens, bi_tokens, pos_tag_list, senti_scores, upper = parse(x_test)
	pos_tags = update({}, pos_tag_list)
	list_ = (uni_tokens, bi_tokens, pos_tags, senti_scores, upper)

	x_test = generateFeatures(list_, unigrams, bigrams, all_pos_tags)
	x_test = [x_test]
	for clf_name, clf in classifiers.items():
		if hasattr(clf, 'predict_prob'):
			pred = clf.predict_proba(x_test)
		else: pred = clf.predict(x_test)
		print clf_name, ': ', pred[0]

def train(X, y):
	test_size = 0.2
	seed = 42
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
	# To evaluate multiple classifiers, store classifiers into a dictionary:
	classifiers = dict() 
	classifiers['GaussianNB'] = GaussianNB()
	classifiers['DecisionTreeClassifier'] = DecisionTreeClassifier(random_state=seed)
	classifiers['SVM'] = SVC()	
	classifiers['LinearSVM'] = LinearSVC()	
	classifiers['MLPClassifier'] = MLPClassifier()	
	classifiers['Perceptron'] = Perceptron()
	classifiers['SGDClassifier'] = SGDClassifier()
	classifiers['KNeighbors Classifier'] = KNeighborsClassifier()
	classifiers['RandomForestClassifier'] = RandomForestClassifier()
	classifiers['BernoulliNB'] = BernoulliNB()
	classifiers['MultinomialNB'] = MultinomialNB()

	# Iterate over dictionary
	for clf_name, clf in classifiers.items(): #clf_name is the key, clf is the value
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)
		score = metrics.accuracy_score(y_test, pred)
		print(clf_name + ': ' + str(score))
		print(metrics.classification_report(y_test, pred))
		print(metrics.confusion_matrix(y_test, pred))

		model_file = cwd+'/'+model_dir+'/'+clf_name+'.pkl'
		joblib.dump(clf, model_file) 

def prepareNgramDicts(raw):
	unigrams = set([])
	all_pos_tags = set([])
	bigrams_dict = dict()
	text = dict()

	counter = 0
	for class_ in classes: text[class_] = []
	for raw_content, class_ in raw.items():
		if counter % 100 == 0 : print 'Processing ', counter, 'of',  len(raw) ,'...'
		uni_tokens, bi_tokens, pos_tag_list, senti_scores, upper = parse(raw_content)
		pos_tags = update({}, pos_tag_list)
		unigrams.update(uni_tokens)
		all_pos_tags.update(pos_tag_list)
		bigrams_dict = update(bigrams_dict, bi_tokens)
		text[class_].append((uni_tokens, bi_tokens, pos_tags, senti_scores, upper))
		counter += 1
			
	bigrams_sorted = sorted(bigrams_dict.items(), key=operator.itemgetter(1), reverse=True)
	bigrams = [tuple_[0] for tuple_ in bigrams_sorted]
	bigrams = bigrams[:1000]
	unigrams, all_pos_tags = list(unigrams), list(all_pos_tags)

	json_file = cwd+'\\'+persist+'.json'
	with open(json_file, 'w') as outfile:
		json.dump((unigrams, bigrams, all_pos_tags), outfile)
	return text


def prepareTraining(text):
	X, y = [], []
	
	unigrams, bigrams, all_pos_tags = loadData()
	counter = 0
	for class_, text_files in text.iteritems():
		for text_file in text_files:
			if counter % 100 == 0 : print 'Processing ', counter, '...'
			x = generateFeatures(text_file, unigrams, bigrams, all_pos_tags)
			X.append(x)
			y.append(class_)
			counter += 1
	print "Commence feature scaling..."
	scaler = preprocessing.MinMaxScaler()
	X = scaler.fit_transform(X)
	return X, y

def generateFeatures(list_, unigrams, bigrams, all_pos_tags):
	x = []
	x.extend(binarize(unigrams, list_[0]))
	x.extend(binarize(bigrams, list_[1]))
	x.extend(binarize(all_pos_tags, list_[2]))
	for senti in list_[3]: x.append(senti)
	x.append(list_[4])
	return x

def sentimentAnalysis(raw_content):
	sid = SentimentIntensityAnalyzer()
	sentiment = sid.polarity_scores(raw_content)
	senti_scores = [score for senti, score in sentiment.items()]
	return senti_scores

def parse(raw_content):
	upper = 0
	senti_scores = sentimentAnalysis(raw_content)
	uni_tokens = nltk.word_tokenize(raw_content)
	pos_tag_tuples = nltk.pos_tag(uni_tokens)
		
	for token in uni_tokens:
		if token.isupper(): upper += 1
	uni_tokens = [token.lower() for token in uni_tokens]
	uni_tokens = [stem(token) for token in uni_tokens]
	uni_tokens = [token for token in uni_tokens if token.isalpha()]
		
	uni_tokens_no_stop = [token for token in uni_tokens if token not in stop_words]
	bi_tokens = list(bigrams(uni_tokens_no_stop))
		
	pos_tag_list = [pos for (token, pos) in pos_tag_tuples]
	return uni_tokens, bi_tokens, pos_tag_list, senti_scores, upper

def fetchTweets():
	print 'Fetching tweets...'
	TWITTER_APP_KEY = 'BFnY5yi7d9P8AzAKjEjTROChp' #supply the appropriate value
	TWITTER_APP_KEY_SECRET = 'Ek4X4NOBl3q2ZB5yviA7mzs9BzDNuz2D1cKcIgEYIO9GVx98JX' 
	TWITTER_ACCESS_TOKEN = '860681898397454336-Glpxs6tzLmXBTL70BcYL1LWsfPJY8hB'
	TWITTER_ACCESS_TOKEN_SECRET = 'ktx83IhNMzeXV3moFgutInSIN1guNQ418kZdy0kCCbEVC'

	twitter = Twython(app_key=TWITTER_APP_KEY, 
	            app_secret=TWITTER_APP_KEY_SECRET, 
	            oauth_token=TWITTER_ACCESS_TOKEN, 
	            oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

	max_len = 5000
	max_id = 0
	tweets = []
	ids = []
	search_results = twitter.search(q='#sarcasm', lang='en', count=100)
	data = search_results['statuses']
	for i, tweet in enumerate(data):
		ids.append(tweet['id'])
		tweets.append(tweet['text'].encode('ascii', 'ignore'))
	max_id = min(ids)

	while len(tweets) < max_len:
		ids = []
		search_results = twitter.search(q='#sarcasm', lang='en', max_id=max_id, count=100)
		data = search_results['statuses']
		for i, tweet in enumerate(data):
			ids.append(tweet['id'])
		max_id = min(ids)
	return tweets

def corpus1(folder):
	raw = {}
	for class_ in classes:
		dir_ = folder+'/'+class_
		for file in os.listdir(dir_):
			if file.endswith('.txt'):
				filename = dir_+'/'+file
				with open(filename) as file:
					raw_content = file.readlines()[0]
					raw[raw_content] = class_
	return raw

def corpus2(folder):
	raw = {}
	corpus2_filename = 'sarcasm_v2.csv'
	dir_ = folder+'/'+corpus2_filename
	df = pd.read_csv(dir_, header=0) 
	for index, row in df.iterrows():
		class_ = row['Label']
		raw_content = row['Response Text']
		raw[raw_content] = class_
	return raw

def loadClassifiers():
	classifiers = dict()
	for file in os.listdir(cwd+'/'+model_dir):
		clf = joblib.load(cwd+'/'+model_dir+'/'+file) 
		classifiers[file] = clf
	return classifiers

def loadData():
	print "Loading data..."
	json_file = cwd+'/'+persist+'.json'
	with open(json_file) as data_file:
		data = json.load(data_file)

	unigrams, bigrams, all_pos_tags = list(data[0]), list(data[1]), list(data[2])
	bigrams = [tuple(bigram) for bigram in bigrams]
	print "Data successfully loaded."
	return unigrams, bigrams, all_pos_tags

def binarize(list_, text_):
	x = []
	for item in list_:
		if item in text_: x.append(1)
		else: x.append(0)
	return x

def update(dict_, list_):
	for item in list_:
		if item in dict_: dict_[item] += 1
		else: dict_[item] = 1
	return dict_

def main():
	folder = 'dataset'
	raw = dict()

	while True:
		inp = raw_input("Would you like to train or test [train/test]: ")
		if inp == 'train':
			print "Reading text files..."
			raw.update(corpus1(folder))
			raw.update(corpus2(folder))

			print 'Preparing dictionaries...'
			text = prepareNgramDicts(raw)

			print "Generating features..."
			X, y = prepareTraining(text)
			
			print "Training model..."
			train(X,y)

		else:
			raw_content = raw_input("Input text to classify: ")
			classifiers = loadClassifiers()
			test(classifiers, raw_content)


if __name__ == '__main__':
    main()