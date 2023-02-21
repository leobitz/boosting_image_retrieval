import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import h5py
import random
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
import proc_lib as proc
import train_lib as trainlib
import models
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_type", type=str)
parser.add_argument("-v", "--vec_type", type=str)
parser.add_argument("-l", "--learning_rate", type=float)
parser.add_argument("-r", "--run", type=int)
args = parser.parse_args()


train_vec_name = "vecs/train-{0}.h5".format(args.vec_type) 
test_vec_name = "vecs/test-{0}.h5".format(args.vec_type)
model_name = args.model_type
vec_out_size = -1

save_folder = "{0}-{1}-{2}-{3}".format(model_name, args.vec_type, args.learning_rate, args.run)
save_path = "result/saves/{0}-{1}-{2}-{3}".format(model_name, args.vec_type, args.learning_rate, args.run)
Path(save_path).mkdir(parents=True, exist_ok=True)

# Load image lables
label_file_path = 'data/label.csv'
images_path = 'data/images/'
df = pd.read_csv(label_file_path)

uniques = []
for col in list( df.columns)[1:-1]:
    uniques.append(list(df[col].unique()))

tt_data, ts_data, vocab, word2id, id2word = proc.get_tag_data(label_file_path)

lines = list(df.iloc[:, 1:-1].itertuples(index=False, name=None))
train_names = [str(x) for x in df.iloc[:3000,0]]
test_names = [str(x) for x in df.iloc[3000:,0]]

outs = proc.prepare_outputs(lines, uniques)
train_outs = [np.array(s[:3000]) for s in outs]
test_outs = [np.array(s[3000:]) for s in outs]
del train_outs[6]
del test_outs[6]
out_dims = [o.shape[1] for o in train_outs]

# image fetaure vector loading from H5 file

f = h5py.File(test_vec_name, 'r')
vecs = f['tensor'][:]
vec_input_size = vecs.shape[1]
if vec_out_size == -1:
    vec_out_size = vec_input_size
f.close()

batch_size = 50
device='cuda'
gen = trainlib.generator("vecs/proc-test-images.h5", test_vec_name, test_outs, ts_data, batch_size=batch_size, rand=False)

name2model = {
    "single": models.SingleNet,
    "double": models.DoubleNet
}
model = name2model[model_name]
net = model(vec_input_size, vec_out_size, out_dims).to(device)
net = net.apply(models.init_weights)

model_saved_names = os.listdir(save_path)
file = open("result/evals/"+save_folder, mode='w')
max_model = None
max_acc = 0
for model_name in model_saved_names:

    net.load_state_dict(torch.load("{0}/{1}".format(save_path, model_name)))
    net.eval()

    embs = []
    # extract the new feature vectors for testing
    n_tbatches = 1000 // batch_size
    net.eval()
    with torch.no_grad():
        for ix in range(n_tbatches):
            images, tags, image_vecs,  labels = next(gen)

            vecs = net.embs(image_vecs)
            
            embs.append(vecs.detach().cpu().numpy())

    embs = np.vstack(embs)
    embsx = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    psim = embsx.dot(embsx.T)

    f = h5py.File("vecs/test-sim.h5", 'r')
    sim_mat = f['tensor'][:]
    f.close()

    idx2map = {}
    map2idx = {}
    for a in range(len(test_names)):
        idx2map[test_names[a]] = a
        map2idx[a] = test_names[a]
    score = proc.get_acc(psim, test_names, map2idx, idx2map, sim_mat)
    line = "{0} {1}".format(model_name, score)
    file.write(line)
    file.write("\n")

    if score > max_acc:
        max_acc = score
        print(max_acc)
        max_model = embs

file.close()

import h5py
f = h5py.File("best.h5", "w")
dset = f.create_dataset("tensor",  data=max_model, dtype='f')

f.close()    