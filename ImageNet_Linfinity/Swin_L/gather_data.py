from utils_sparse import read_list_with_json

import glob

pattern = 'results/*_Benchmark_*.pt'

# Use glob to find all files that match the pattern
files = glob.glob(pattern)

# Print the list of files that match the pattern
rb_acc = 0
acc = 0
counter = 0
for file in files:
    data = read_list_with_json(file)
    counter += 1
    rb_acc += data['Robust_accuracy']
    acc += data['top1']
rb_acc /= counter
acc /= counter
print(rb_acc)
print(acc)
