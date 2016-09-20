# -*- coding: utf-8 -*-
__author__ = 'Chason'

import csv
import numpy as np
from sklearn import svm
import cPickle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

def load_dataset(train_filename, test_filename):
    print "Loading data...",
    train_reader = csv.reader(file(train_filename, 'rb'))
    test_reader = csv.reader(file(test_filename, 'rb'))
    print "Done!"

    train_x = []
    train_y = []
    test_x = []
	
    print "Transforming data...",
	
    header = True
    for line in train_reader:
        if header == False:
            tmp = []
            for c in line[1:]:
                tmp.append(int(c)/255.0)
            train_x.append(tmp)
            train_y.append(int(line[0]))
        else:
            header = False
			
    header = True
    for line in test_reader:
        if header == False:
            tmp = []
            for c in line:
                tmp.append(int(c)/255.0)
            test_x.append(tmp)
        else:
            header = False
			
    print "Done!"
    return train_x, train_y, test_x

def svm_train(train_x, train_y, filename):
    svm_model = svm.SVC()
    print "fitting svm model...",
    svm_model = svm_model.fit(train_x, train_y)
    print "svm model fitted!"
    print svm_model
    print
    print "saving svm model...",
    f = open(filename, "wb")
    cPickle.dump(svm_model, f, -1)
    f.close()
    print "svm model saved!"
    return svm_model

def load_model(filename):
    print "Loading svm model...",
    f = open(filename, "rb")
    svm_model = cPickle.load(f)
    f.close()
    print "svm model loaded!"
    return svm_model
	
def calc_train_performance(svm_model, train_x, train_y):
    print "predicting train dataset...",
    train_pred = svm_model.predict(train_x)
    print "done!"
    print "train_pred =", train_pred

    train_perf = 0.0
    for i, p in enumerate(train_pred):
        if p == train_y[i]:
            train_perf += 1
    train_perf /= len(train_y)

    log = 'train perf %f %%' % (train_perf * 100)
    print log
    f = open("svm model.log", "w")
    f.write(log)
    f.close()

def get_test_prediction(svm_model, test_x):
    print "predicting test dataset...",
    test_pred = svm_model.predict(test_x)
    print "done!"
    print "test_pred =", test_pred
    return test_pred

def output_test(test_y):
    with open('submission.csv', 'wb') as csvfile:
        submitwriter = csv.writer(csvfile)
        submitwriter.writerow(['ImageId', 'Label'])
        image_id = 1
        for t in test_y:
            submitwriter.writerow([image_id, t])
            image_id += 1
        print "submission saved."

def main_train_model():
	train_x, train_y, test_x = load_dataset("train.csv", "test.csv")
	svm_model = svm_train(train_x, train_y, "svm.model")
	calc_train_performance(svm_model, train_x, train_y)
	test_y = get_test_prediction(svm_model, test_x)
	output_test(test_y)

def main_load_model():
	train_x, train_y, test_x = load_dataset("train.csv", "test.csv")
	svm_model = load_model("svm.model")
	calc_train_performance(svm_model, train_x, train_y)
	test_y = get_test_prediction(svm_model, test_x)
	output_test(test_y)

# main_train_model()
main_load_model()
