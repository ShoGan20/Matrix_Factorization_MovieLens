import csv
import pickle
import random
import numpy as np
from Mat_Fact import Matrix_Fact

in_testcsv = 'ml-latest-small/testRatings.csv'
out_model = 'trainedModel.pkl'

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

def writecsv(rec, fname):
	print('Writing ' + fname)
	with open(fname, 'w') as csv_file:
		csv_file.write('Header\n')
		for r in rec:
			csv_file.write(str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + '\n')
	print('Done Writing!')

def main():
	print('Reading from ' + out_model)
	mat_fact = 0
	with open(out_model, 'rb') as input:
		mat_fact = pickle.load(input)
	print('Done reading from ' + out_model)

	mse = 0
	mae = 0
	testSubset = readcsv(in_testcsv)
	for r in testSubset:
		mse += pow(r[2] - mat_fact.rating(r[0], r[1]), 2)
		mae += abs(r[2] - mat_fact.rating(r[0], r[1]))
	mse /= len(testSubset)
	mae /= len(testSubset)

	print('Train mean square error = ' + str(mat_fact .MSE()))
	print('Test mean square error = ' + str(mse))
	print('Test mean absolute error = ' + str(mae))

	print('\nSample values:')
	print('Expected','Predicted')
	random.seed(0)
	for r in random.sample(testSubset, 100):
		print(r[2], mat_fact.rating(r[0], r[1]))
