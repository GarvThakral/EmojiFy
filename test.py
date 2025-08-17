import numpy as np
import pandas as pd
import json
with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    index = 1
    word_to_index_map = {}
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_to_index_map[word] = index

        index+=1
with open("test.json",'w') as f:
    f.writelines(json.dumps(word_to_index_map))

with open("test.json" , 'r') as f:
    lines = f.readlines()
    print(type(lines[0]))
    print("Starting ...")
    print(lines[0][:10])
    print(lines[0][-10:])
    print("...ending")
    word_to_index_map = json.loads(lines[0])
    print(word_to_index_map['the'])
