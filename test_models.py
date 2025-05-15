from datasets import Dataset, load_dataset
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from fact_checking import load_base
from fact_checking.bert import check_claim_bert
from fact_checking.lstm import LSTM, vocab, check_claim_lstm
from fact_checking.rnn import check_claim_rnn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------Przygotowanie danych----------------#

test_dataset = load_dataset("pietrolesci/nli_fever")
test_dataset = test_dataset["dev"]
test_dataset = test_dataset.shuffle()
X = test_dataset["premise"]
y = test_dataset["label"]


# # print(X)
# # print(y)
count = 200
label_map = {0: 0, 2: 1, 1: 2}
y = [label_map[item] for item in y]

y = y[:count]
X = X[:count]

# ----------------Prosta pętla klasyfikująca---------------- #
# y_rnn = []
# y_bert = []
# y_lstm = []
# for i in tqdm(range(count)):
#     y_rnn.append(check_claim_rnn(X[i]))
#     y_bert.append(check_claim_bert(X[i]))
#     y_lstm.append(check_claim_lstm(X[i]))

# ----------------Walidacja krzyżowa---------------- #

kf = KFold(n_splits=10)
acc_bert = []
acc_lstm = []
acc_rnn = []

for train_idx, test_idx in tqdm(kf.split(X)):
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

#    y_bert = [check_claim_bert(x) for x in X_test]
    y_lstm = [check_claim_lstm(x) for x in X_test]
#    y_rnn = [check_claim_rnn(x) for x in X_test]

#    acc_bert.append(accuracy_score(y_test, y_bert))
    acc_lstm.append(accuracy_score(y_test, y_lstm))
#    acc_rnn.append(accuracy_score(y_test, y_rnn))


#with open("y_bert", "wb") as fp:
#    pickle.dump(y_bert, fp)
with open("y_lstm", "wb") as fp:
    pickle.dump(y_lstm, fp)
#with open("y_rnn", "wb") as fp:
#    pickle.dump(y_rnn, fp)
#with open("y_bert", "rb") as fp:
#    y_bert_t = pickle.load(fp)
with open("y_lstm", "rb") as fp:
    y_lstm = pickle.load(fp)
with open("y_rnn", "rb") as fp:
    y_rnn = pickle.load(fp)
#print("bert: ", y_bert)
print("lstm: ", y_lstm)
#print("rnn: ", y_rnn)
print("true: ", y)
# ----------------Test McNemara----------------#

# from statsmodels.stats.contingency_tables import mcnemar
#
# correct_bert = np.array(y_bert) == np.array(y)
# correct_lstm = np.array(y_lstm) == np.array(y)
#
# both_correct = np.sum((correct_bert == True) & (correct_lstm == True))
# bert_only = np.sum((correct_bert == True) & (correct_lstm == False))
# lstm_only = np.sum((correct_bert == False) & (correct_lstm == True))
# both_false = np.sum((correct_bert == False) & (correct_lstm == False))
#
# table = [[both_correct, bert_only], [lstm_only, both_false]]
# print(table)
# result = mcnemar(table, exact=True)
# print("Test McNemara:")
# print("p-value: ", result.pvalue)
#
# if result.pvalue < 0.05:
#     print("Różnica jest statystycznie istotna")
# else:
#     print("Różnica nie jest statystycznie istotna")
#

# ----------------Test Wilcoxona----------------#

from scipy.stats import wilcoxon

#print("wynik bert: ", acc_bert)
print("wynik lstm: ", acc_lstm)

#stat, p = wilcoxon(acc_bert, acc_lstm)
#print("Test Wilcoxona:")
#print("p_value: ", p)
#if p < 0.05:
#    print("Różnica jest statystyczne istotna")
#else:
#    print("Różnica jest statystycznie nieistotna")
