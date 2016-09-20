# -*- coding: utf-8 -*-
__author__ = 'Chason'

from svm_model import *
import sys

if __name__ == '__main__':
	argc = len(sys.argv)
	if argc == 1:
		main_load_model("svm.model")
	elif argc == 2:
		main_load_model(sys.argv[1])
	else:
		print "Usage: python load_svm_model [your model filename(default:'svm.model')]"
		

