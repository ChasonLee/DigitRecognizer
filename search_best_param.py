# -*- coding: utf-8 -*-
__author__ = 'Chason'

from svm_model import *
import sys

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc == 1:
        # test()
        main_search_best_param("best_svm.model")
    elif argc == 2:
        main_search_best_param(sys.argv[1])
    else:
        print "Usage: python search_best_param.py [your model filename(default:'best_svm.model')]"
