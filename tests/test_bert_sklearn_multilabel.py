import os
import sys
import csv
import shutil

import pytest
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

from bert_sklearn import BertClassifier
from bert_sklearn import load_model

DATADIR = 'tests/data'

def setup_function(function):
    print ("\n" + "="*75)


def teardown_function(function):
    print ("")


def toxic_test_data(train_file=DATADIR + '/toxic/train.csv',
                   dev_file=DATADIR + '/toxic/test.csv'):

    label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    train = pd.read_csv(train_file, encoding='utf8')
    X_train = train["comment_text"].fillna("DUMMY_VALUE").values.tolist()
    y_train = train[label_list].values.tolist()

    dev = pd.read_csv(train_file, encoding='utf8')
    X_dev = dev["comment_text"].fillna("DUMMY_VALUE").values.tolist()
    y_dev = dev[label_list].values.tolist()

    # print('X_train:', X_train)
    # print('y_train:', y_train)
    # print('X_dev:', X_dev)
    # print('y_dev:', y_dev)
    return X_train, y_train, X_dev, y_dev, label_list


def test_bert_sklearn_accy():
    """
    Test bert_sklearn accuracy
    compare against  huggingface run_classifier.py
    on 200 rows of SST-2 data.
    """
    print("Running bert-sklearn...")
    X_train, y_train, X_dev, y_dev, label_list = toxic_test_data()

    # define model
    model = BertClassifier()
    model.validation_fraction = 0.0
    model.learning_rate = 5e-5
    model.gradient_accumulation_steps = 2
    model.max_seq_length = 64
    model.train_batch_size = 16
    model.eval_batch_size = 8
    model.epochs = 2
    model.multilabel = True # for multi-label classification
    model.label_list = label_list

    model.fit(X_train, y_train)

    bert_sklearn_accy = model.score(X_dev, y_dev)
    bert_sklearn_accy /= 100

    # run huggingface BERT run_classifier and check we get the same accuracy
    cmd = r"python tests/run_classifier.py --task_name sst-2 \
                                --data_dir ./tests/data/sst2 \
                                --do_train  --do_eval \
                                --output_dir ./comptest \
                                --bert_model bert-base-uncased \
                                --do_lower_case \
                                --learning_rate 5e-5 \
                                --gradient_accumulation_steps 2 \
                                --max_seq_length 64 \
                                --train_batch_size 16 \
                                --eval_batch_size 8 \
                                --num_train_epochs 2"

    print("\nRunning huggingface run_classifier.py...\n")
    os.system(cmd)
    print("...finished run_classifier.py\n")

    # parse run_classifier.py output file and find the accy
    accy = open("comptest/eval_results.txt").read().split("\n")[0] # 'acc = 0.76'
    accy = accy.split("=")[1]
    accy = float(accy)
    print("bert_sklearn accy: %.02f, run_classifier.py accy : %0.02f"%(bert_sklearn_accy, accy))

    # clean up
    print("\nCleaning up eval file: eval_results.txt")
    #os.remove("eval_results.txt")
    shutil.rmtree("comptest")
    assert bert_sklearn_accy == accy


def test_save_load_model():
    """Test saving/loading a fitted model to disk"""

    X_train, y_train, X_dev, y_dev, label_list = toxic_test_data()

    model = BertClassifier()
    model.max_seq_length = 64
    model.train_batch_size = 8
    model.epochs= 1
    model.multilabel = True
    model.label_list = label_list

    model.fit(X_train, y_train)

    accy1 = model.score(X_dev, y_dev)

    savefile='./test_model_save.bin'
    print("\nSaving model to ", savefile)

    model.save(savefile)

    # load model from disk
    new_model = load_model(savefile)

    # predict with new model
    accy2 = new_model.score(X_dev, y_dev )

    # clean up
    print("Cleaning up model file: test_model_save.bin ")
    os.remove(savefile)

    assert accy1 == accy2


def test_not_fitted_exception():
    """Test predicting with a model that has not been fitted"""

    X_train, y_train, X_dev, y_dev, label_list = toxic_test_data()

    model = BertClassifier()
    model.max_seq_length = 64
    model.train_batch_size = 8
    model.epochs= 1
    model.multilabel = True
    model.label_list = label_list

    # model has not been fitted: model.fit(X_train, y_train)
    with pytest.raises(Exception):
        model.score(X_dev, y_dev)
