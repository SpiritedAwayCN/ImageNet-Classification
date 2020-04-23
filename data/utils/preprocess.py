import csv
import os
import shutil

folders = 'train',

for fn in folders:
    print('moving', fn, '...')
    with open('{}.csv'.format(fn), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "filename":
                continue
            label = row[1]
            path = os.path.join(fn, label, row[0])
            
            if not os.path.isfile(path):
                print(path)
                exit(0)