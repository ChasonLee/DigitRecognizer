# -*- coding: utf-8 -*-
__author__ = 'Chason'

import csv
import numpy as np
from sklearn import svm
import cPickle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import time
import random

def load_dataset(train_filename, test_filename):
    print "Loading data...",
    train_reader = csv.reader(file(train_filename, 'rb'))
    test_reader = csv.reader(file(test_filename, 'rb'))
    print "Done!"

    train_x = []
    train_y = []
    test_x = []

    print "Transforming data...",

    start = time.clock()
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

    end = time.clock()
    print "Done!\tused time:%s s"%time.strftime('%H:%M:%S',time.gmtime(end - start))
    return train_x, train_y, test_x

def svm_train(train_x, train_y, filename = None, C = None, gamma = None):
    if C == gamma == None:
        svm_model = svm.SVC()
    else:
        svm_model = svm.SVC(C=C, gamma=gamma)
    print "Fitting SVM model...",
    start = time.clock()
    svm_model = svm_model.fit(train_x, train_y)
    end = time.clock()
    print "SVM model fitted!\tused time:%s s"%time.strftime('%H:%M:%S',time.gmtime(end - start))
    print svm_model
    if filename != None:
        print "Saving svm model...",
        f = open(filename, "wb")
        cPickle.dump(svm_model, f, -1)
        f.close()
        print "SVM model saved!"
    return svm_model

def load_model(filename):
    print "Loading svm model...",
    f = open(filename, "rb")
    svm_model = cPickle.load(f)
    f.close()
    print "SVM model loaded!"
    return svm_model


def calc_train_performance(svm_model, train_x, train_y):
    print "Predicting train dataset...",
    start = time.clock()
    train_pred = svm_model.predict(train_x)
    end = time.clock()
    print "Done!\tused time:%s s"%time.strftime('%H:%M:%S',time.gmtime(end - start))
    # print "train_pred =", train_pred

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
    return train_perf

def calc_cv_performance(svm_model, cv_x, cv_y):
    print "Predicting CV dataset...",
    start = time.clock()
    cv_pred = svm_model.predict(cv_x)
    end = time.clock()
    print "Done!\tused time:%s s"%time.strftime('%H:%M:%S',time.gmtime(end - start))
    # print "cv_pred =", cv_pred
    cv_perf = 0.0
    for i, p in enumerate(cv_pred):
        if p == cv_y[i]:
            cv_perf += 1
    cv_perf /= len(cv_y)
    log = 'cv perf %f %%' % (cv_perf * 100)
    print log
    # f = open("svm model.log", "w")
    # f.write(log)
    # f.close()
    return cv_perf

def get_test_prediction(svm_model, test_x):
    print "Predicting test dataset...",
    start = time.clock()
    test_pred = svm_model.predict(test_x)
    end = time.clock()
    print "Done!\tused time:%s s"%time.strftime('%H:%M:%S',time.gmtime(end - start))
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

def get_cross_validation(train_x, train_y):
    cv_x = []
    cv_y = []
    print "selecting cross validation...",
    train_len = len(train_x)
    for i in range(4000):
        inx = int(random.random()*train_len)
        cv_x.append(train_x[inx])
        cv_y.append(train_y[inx])
        del train_x[inx]
        del train_y[inx]
        train_len -= 1
    print "Done!"
    print len(train_x), len(train_y), len(cv_x), len(cv_y)
    return cv_x, cv_y
	
def main_train_model(model_filename, C = None, gamma = None):
    train_x, train_y, test_x = load_dataset("train.csv", "test.csv")
    svm_model = svm_train(train_x, train_y, model_filename, C, gamma)
    calc_train_performance(svm_model, train_x, train_y)
    test_y = get_test_prediction(svm_model, test_x)
    output_test(test_y)
	
def main_load_model(model_filename):
    train_x, train_y, test_x = load_dataset("train.csv", "test.csv")
    svm_model = load_model(model_filename)
    calc_train_performance(svm_model, train_x, train_y)
    test_y = get_test_prediction(svm_model, test_x)
    output_test(test_y)

def main_search_best_param(model_filename):
    train_x, train_y, test_x = load_dataset("train.csv", "test.csv")
    cv_x, cv_y = get_cross_validation(train_x, train_y)
    C_range = np.logspace(-6, 8, 15)
    # gamma_range = np.logspace(-6, 8, 10)
    print "C range:", C_range
    best_cv_perf = 0
    for c in C_range:
        print
        print "Testing [C = %f, gamma = 'auto']:"%(c)
        svm_model = svm_train(train_x=train_x, train_y=train_y, C=c, gamma='auto')
        cv_perf = calc_cv_performance(svm_model, cv_x, cv_y)
        if cv_perf > best_cv_perf:
            f = open(model_filename, "wb")
            cPickle.dump(svm_model, f, -1)
            f.close()
            print "Best param for now! Model saved."
            cv_perf = best_cv_perf

def test():
    train_x, train_y, test_x = load_dataset("train.csv", "test.csv")
    svm_model = svm_train(train_x=train_x, train_y=train_y, C=0.1, gamma='auto')