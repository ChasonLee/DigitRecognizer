# -*- coding: utf-8 -*-
__author__ = 'Chason'

from svm_model import *
import sys

if __name__ == '__main__':
	argc = len(sys.argv)
	if argc == 1:
		main_train_model("svm.model")
	elif argc == 2:
		main_train_model(sys.argv[1])
	else:
		print "Usage: python train_svm_model.py [your model filename(default:'svm.model')]"
