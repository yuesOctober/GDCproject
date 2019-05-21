# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from utils import logger
from hubgenes import hubSelection
#def lassoSelection(X,y,)
def extraTreeSelection(X_train,y_train,n):
	forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
	forest.fit(X_train, y_train)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]
	features = [indices[i] for i in range(n)]
	return features
def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=-2)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	print(n_features)
	print (n)
	#print(n_features)
	while n_features > n:
		print(sfm.threshold)
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
		print(n_features)
	print (n_features)
	features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
	logger.info("selected features are {}".format(features))
	return features
def varianceSelection(x_train,y_train):

	selector = VarianceThreshold(1.0)
	selector.fit(x_train)
	features = selector.get_support(True)
	return features
def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
	real_negative = len(y_true) - sum(y_true)
	return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(2018)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	models = {
		'LogisticRegression': LogisticRegression(),
		'ExtraTreesClassifier': ExtraTreesClassifier(),
		'RandomForestClassifier': RandomForestClassifier(),
    	'AdaBoostClassifier': AdaBoostClassifier(),
    	'GradientBoostingClassifier': GradientBoostingClassifier(),
    	'SVC': SVC()
	}
	tuned_parameters = {
		'LogisticRegression':{'C': [1, 10]},
		'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		'RandomForestClassifier': { 'n_estimators': [16, 32] },
    	'AdaBoostClassifier': { 'n_estimators': [16, 32] },
    	'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    	'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	}
	scores= {}
	for key in models:
		clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
		clf.fit(X_train,y_train)
		y_test_predict = clf.predict(X_test)
		precision = precision_score(y_test, y_test_predict)
		accuracy = accuracy_score(y_test, y_test_predict)
		f1 = f1_score(y_test, y_test_predict)
		recall = recall_score(y_test, y_test_predict)
		specificity = specificity_score(y_test, y_test_predict)
		scores[key] = [precision,accuracy,f1,recall,specificity]
	#print(scores)
	return scores



def draw(scores):
	'''
	draw scores.
	'''
	import matplotlib.pyplot as plt
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	categories = []
	specificities = []
	N = len(scores)
	ind = np.arange(N)  # set the x locations for the groups
	width = 0.1        # the width of the bars
	for key in scores:
		categories.append(key)
		precisions.append(scores[key][0])
		accuracies.append(scores[key][1])
		f1_scores.append(scores[key][2])
		recalls.append(scores[key][3])
		specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
	specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	print(categories)
	ax.set_xticks(np.arange(N))
	ax.set_xticklabels(categories)
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.grid()
	plt.show()






def getHubFeatures(X_train,y_train):
	n = 50
	X_train_tumor= []
	lasso_features_columns = extraTreeSelection(X_train, y_train, n)
	feature_mapping = {}
	for i in range(len(lasso_features_columns)):
		feature_mapping[i]=lasso_features_columns[i]

	for index in range(len(y_train)):
		if y_train[index] ==0:
			X_train_tumor.append(X_train[index])
	total_tumor = len(X_train_tumor)
	mixed = int(total_tumor*0.1)
	count = 0
	for index in range(len(y_train)):

		if (y_train[index] ==1 & count<=mixed):
			X_train_tumor.append(X_train[index])
			count+=1
	print (X_train_tumor)
	X_train_tumor = np.array(X_train_tumor)
	# features_columns = lasso_features_columns
	# hub_feature_columns = features_columns
	#hub selection
	features_columns = hubSelection(X_train_tumor[:,lasso_features_columns])
	
	print (features_columns)
	hub_feature_columns=[]
	for feature in features_columns:
		hub_feature_columns.append(lasso_features_columns[feature])
	return hub_feature_columns

def genSubBatches(X_train,y_train,percentage=0.2,copies=5):
	tumor_index = np.where(y_train==1)[0]
	normal_index = np.where(y_train==0)[0]
	tumor_count = len(tumor_index)
	normal_count = len(normal_index)
	picked_tumor_count = (int)(tumor_count*percentage)
	picked_normal_count = (int)(normal_count*percentage)

	batches = []
	for i in range(copies):
		picked_tumor_index = tumor_index[np.random.choice(tumor_count, picked_tumor_count, replace=False) ]
		picked_normal_index= normal_index[np.random.choice(normal_count, picked_normal_count, replace=False)]
		picked_index = np.concatenate((picked_tumor_index,picked_normal_index))
		batches.append([X_train[picked_index],y_train[picked_index]])
	return batches

if __name__ == '__main__':


	data_dir ="/Users/yueshi/Downloads/GDCproject/data/"

	# data_file = data_dir + "miRNA_matrix.csv"
	data_file = data_dir + "merged_miRNA.csv"

	df = pd.read_csv(data_file)
	# print(df)
	y_data = df.pop('label').values

	df.pop('file_id')

	columns =df.columns
	#print (columns)
	X_data = df.values
	
	# split the data to train and test set
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.7, random_state=0)
	train_counts = len(X_train)
	top_percentage = 1
	#select certain samples :
	nselected = (int)(top_percentage*train_counts)
	X_train  = X_train[:nselected]
	y_train = y_train[:nselected]

	print(y_train)

	#standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# check the distribution of tumor and normal sampels in traing and test data set.
	logger.info("Percentage of tumor cases in training set is {}".format(float(sum(y_train))/float(len(y_train))))
	logger.info("Percentage of tumor cases in test set is {}".format(float(sum(y_test))/float(len(y_test))))
	
	total_batches = 2
	batches = genSubBatches(X_train,y_train,copies = total_batches,percentage=0.2)

	total_hub_feature_columns = getHubFeatures(X_train,y_train)
	print ("total_hub_feature_columns are:",total_hub_feature_columns)
	print("hub_feature_columns are {}".format(total_hub_feature_columns))
	hub_genes = []
	for i in range(total_batches):
		print ("batch {} starts".format(i))
		batch_X_train = batches[i][0]
		batch_y_train = batches[i][1]
		batch_hub_features = getHubFeatures(batch_X_train,batch_y_train)
		hub_genes.append(batch_hub_features)
	print ("hub_genes are {}".format(hub_genes))
	# hub_feature_columns = hub_genes[0]
	# final_
	hub_feature_columns = []
	for i in range(total_batches):
		intersection_hub_feature_columns  = list(set(total_hub_feature_columns).intersection(hub_genes[i]))
		hub_feature_columns.extend(intersection_hub_feature_columns)

	print (hub_feature_columns)
	# hub_feature_columns = getHubFeatures(X_train,y_train)

	# n = 50
	# X_train_tumor= []
	# X_train_tumor = np.array([X_train[i] for i in range(len(X_train)) if y_train[i]==1])
	# print (y_train)

	# lasso_features_columns = lassoSelection(X_train, y_train, n)
	# lasso_features_columns = extraTreeSelection(X_train, y_train, n)
	
	# lasso_features_columns = varianceSelection(X_train, y_train)



	# print (lasso_features_columns)
	# print(len(lasso_features_columns))
	# feature_mapping = {}
	# for i in range(len(lasso_features_columns)):
	# 	feature_mapping[i]=lasso_features_columns[i]

	# for index in range(len(y_train)):
	# 	if y_train[index] ==1:
	# 		X_train_tumor.append(X_train[index])


	# X_train_tumor = np.array(X_train_tumor)
	# print (X_train_tumor)



	
	# features_columns = lasso_features_columns
	# feaures_columns = hubSelection(X_train_tumor)

	'''
	hub feature columns
	'''
	# features_columns = hubSelection(X_train_tumor[:,lasso_features_columns])


	# features_columns = [2,22,25,28,45]
	# hub_feature_columns=[]
	# for feature in features_columns:
	# 	hub_feature_columns.append(lasso_features_columns[feature])


	'''
	without hubs
	'''
	# hub_feature_columns = lasso_features_columns


	# hub_feature_columns = lasso_features_columns
	scores = model_fit_predict(X_train[:,hub_feature_columns],X_test[:,hub_feature_columns],y_train,y_test)

	draw(scores)
	#lasso cross validation
	# lassoreg = Lasso(random_state=0)
	# alphas = np.logspace(-4, -0.5, 30)
	# tuned_parameters = [{'alpha': alphas}]
	# n_fold = 10
	# clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
	# clf.fit(X_train,y_train)




 




