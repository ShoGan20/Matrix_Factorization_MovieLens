import urllib.request
from pathlib import Path
import zipfile
import split_train_test
import train
import test
import time

start_time = time.time()

Dataset = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

filename = 'ml-latest-small.zip'

if Path(filename).is_file():
    print(filename + ' already exists!')

else:
    print('Downloading ' +filename)
    urllib.request.urlretrieve(Dataset, filename)

with zipfile.ZipFile(filename, 'r') as myzip:
    myzip.extractall('')

split_train_test.main()

train.main()

test.main()

print("Program finished --- %s seconds ---" % (time.time() - start_time))
