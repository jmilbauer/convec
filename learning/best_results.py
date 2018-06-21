import os
import re
tracker = []

for root, folder, files in os.walk('.'):
    for fname in files:
        if fname.endswith('.txt'):
            name = re.match(r'([0-9])layer_([0-9]+)neuron_([0-9]+)emb_0.txt', fname)
            if name is not None:
                with open(fname, 'r') as fh:
                    text = fh.read()
                match = re.search(r'Test accuracy: ([0-9\.]+)\n', text)
                if match is not None:
                    tracker.append(('{} & {} & {} & {:.2f}\\%\\\\'.format(name.group(3), name.group(1), name.group(2), 100*float(match.group(1))), 100*float(match.group(1))))

tracker = sorted(tracker, key = lambda x: x[1], reverse = True)

for i in range(10):
    print(tracker[i][0])
