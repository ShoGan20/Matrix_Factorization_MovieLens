import csv
import pickle
import random
import threading
import numpy as np
from Mat_Fact import Matrix_Fact

def readcsv(fname):
    print('Reading ' + fname)
    rec = []

    with open(fname) as csv_file:
        csv_read = csv.reader(csv_file, delimiter = ',')
        line_ct = 0

        for r in csv_read:
            if line_ct != 0:
                rec.append([int(r[0]), int(r[1]), float(r[2])])
            line_ct += 1
        
        print('Read ' + str(line_ct) + ' entries')
    print('Done Reading!')
    return rec


def MaxID(csvRec):
    maxid = 0
    maxmovid = 0

    for r in csvRec:
        if r[0] > maxid:
            maxid = r[0]
        if r[1] > maxmovid:
            maxmovid = r[1]
    
    return maxid, maxmovid

def ratings(csvRec, maxid, maxmovid):
    Rate = np.zeros((maxid+1, maxmovid+1), dtype='uint8')
    for r in csvRec:
        Rate[r[0]][r[1]] = r[2]
    
    return Rate

def Training(D, alpha, beta, lmda, epochs, err):
    print('Dimension=' + str(D) + ', alpha=' + str(alpha) + ', beta=' + str(beta) + ', lambda=' + str(lmda) + ', Epochs=' + str(epochs) + ', max error=' + str(err))
    in_csv = 'ml-latest-small/ratings.csv'
    in_testcsv = 'ml-latest-small/trainRatings.csv'
    out_model = 'trainedModel.pkl'

    rating = readcsv(in_csv)
    train_set = readcsv(in_testcsv)
    maxid, maxmovid = MaxID(rating)
    Rate = ratings(train_set, maxid, maxmovid)
    mat_fact = Matrix_Fact(Rate, D, alpha, beta, lmda, epochs, err)

    print('Training ----')
    train_process = mat_fact.train()
    print('Done! MSE = ' + str(mat_fact.MSE()))


    print('Saving model as ' + out_model)
    with open(out_model, 'wb') as output:
        pickle.dump(mat_fact, output, pickle.HIGHEST_PROTOCOL)
    print('Done!')

def main():
    Training(100, 0.06, 0.06, 0.95, 100, 0.0)
