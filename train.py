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

batch_size = 32
device='cuda'
gen = trainlib.generator("vecs/proc-train-images.h5", train_vec_name, train_outs, tt_data, batch_size=batch_size)
name2model = {
    "single": models.SingleNet,
    "double": models.DoubleNet
}
model = name2model[model_name]
net = model(vec_input_size, vec_out_size, out_dims).to(device)
net = net.apply(models.init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate , weight_decay=1e-5)

for epoch in range(2000):  # loop over the dataset multiple times

    running_loss = 0.0
    n_batches = 3000 // batch_size
    net.train()
    epoch_loss = 0
    for i in range(n_batches):
    
        images, tags, image_vecs,  labels = next(gen)
        
        optimizer.zero_grad()

        results = net(images, tags[:, :, 0], image_vecs)
        loss = 0
        
        for lx in range(len(results)):
            tmp_loss = criterion(results[lx], labels[lx])
            loss += tmp_loss
        loss = loss / len(results)

        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % (n_batches // 10) == 0 and i > 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / (n_batches // 10)))
            running_loss = 0.0
            
    print("Epoch loss: " + str(epoch_loss / n_batches))
   
    if epoch % 5 == 0 and epoch > 1:
        model_path = "{0}/{1}-{2}".format(save_path, epoch, str(epoch_loss / n_batches))
        torch.save(net.state_dict(), model_path)

    if epoch_loss / n_batches < 0.001:
        break
print('Finished Training')
model_path = "{0}/{1}-{2}".format(save_path,  "final=" + str(epoch), str(epoch_loss / n_batches))
torch.save(net.state_dict(), model_path)
# model types = 2
# Learning Rate = 2
# Vec types = 5
