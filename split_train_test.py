import csv
import random

split_ratio = 0.9
r_seed = 9
in_csv = 'ml-latest-small/ratings.csv'
out_traincsv = 'ml-latest-small/trainRatings.csv'
out_testcsv = 'ml-latest-small/testRatings.csv'

def readcsv(fname):
	print('Reading ' + fname)
	records = []
	with open(fname) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for r in csv_reader:
			if line_count != 0:
				records.append([ int(r[0]), int(r[1]), float(r[2]) ])
			line_count += 1
		print('Read ' + str(line_count) + ' entries from ' + fname)
	print('Done reading ' + fname)
	return records

def writecsv(records, fname):
	print('Writing ' + fname)
	with open(fname, 'w') as csv_file:
		csv_file.write('Header\n')
		for r in records:
			csv_file.write(str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + '\n')
	print('Done writing ' + fname)

def main():
	ratings = readcsv(in_csv)

	print('Preparing test & train data...')
	random.seed(a=r_seed)
	trainSubset, testSubset = [], []
	for row in ratings:
		if random.uniform(0,1) <= split_ratio:
			trainSubset.append(row)
		else:
			testSubset.append(row)
	print('Done preparing test & train data...')

	writecsv(trainSubset, out_traincsv)
	writecsv(testSubset, out_testcsv)