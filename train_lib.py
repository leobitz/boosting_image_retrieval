import h5py
import torch
import numpy as np
# generator function to generator the inputs and outputs
def generator(vec_name, emb_name, outputs, tag_data,  batch_size=64, rand=True, device='cuda'):
    
    f = h5py.File(vec_name, 'r')
    input_batch = torch.tensor(f['tensor'][:])
    f.close()
    
    f = h5py.File(emb_name, 'r')
    pca_vec = f['tensor'][:]
    pca_vec = torch.tensor(pca_vec).to(device)
    tag_data = torch.LongTensor(tag_data).to(device)
    outputs = [torch.LongTensor(o.argmax(axis=1)).to(device) for o in outputs]
    f.close()
    indexes = np.arange(len(outputs[0]))
    n_batch = len(outputs[0]) // batch_size
    if rand:
        np.random.shuffle(indexes)
    current  = 0
    while True:
        bs = indexes[current:current+batch_size]
        bs_images = input_batch[bs].to(device)
        bt = tag_data[bs]
        bs_vec = pca_vec[bs]
        bs_outs = [o[bs] for o in outputs]
        yield bs_images, bt, bs_vec, bs_outs
        current += batch_size
        if current >= batch_size * n_batch:
            current = 0
            if rand:
                np.random.shuffle(indexes)
            