import os
hidden_layer_size = [150, 300]
num_layers = [1, 2]
embedding_dim = [100, 150]

f1 = open('r1.sh', 'w')
f2 = open('r2.sh', 'w')
ct = 0
for nl in num_layers:
    for hls in hidden_layer_size:
            for ed in embedding_dim:
                for i in range(1):
                    if ct %2:
                        fh = f1
                    else:
                        fh = f2
                    output_file = '{}layer_{}neuron_{}emb_{}.txt'.format(nl, hls, ed, i)
                    if os.path.exists(output_file):
                        continue
                    ct += 1
                    fh.write('echo python3 classifier.py --hidden_layer_size {} --num_layers {} --embedding_dim {} i={}\n'.format(hls, nl, ed, i))
                    fh.write('time python3 classifier.py --hidden_layer_size {} --num_layers {} --embedding_dim {} --output_file {}layer_{}neuron_{}emb_{}.pkl > {}layer_{}neuron_{}emb_{}.txt\n'.format(hls, nl, ed, nl, hls, ed, i, nl, hls, ed, i))
