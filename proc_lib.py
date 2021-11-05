import numpy as np
import torch.nn as nn
import random
from skimage import io
import matplotlib.pyplot as plt
from collections import Counter

def add_props(prop2idx, idx2prop, prop):
    """
    if the property is in prop2idx, returns the index or add a new one and return
    """
    if prop not in prop2idx:
        prop2idx[prop] = len(prop2idx)
        idx2prop[prop2idx[prop]] = prop
    return prop2idx[prop]


def read_annotation(path):
    lines = open(path).readlines()
    lines = lines[1:]
    id2prop = {}
    idx2name = {}
    name2idx = {}
    prop2idx = {}
    idx2prop = {}
    tag2idx = {}
    idx2tag = {}
    name2line = {}
    # name2rawline = {}
    for k in range(len(lines)):
        line = lines[k]
        line = line.lower().strip().split(",")
        props = line[1:-1]  # color, gender, season...
        del props[-2]  # remove the year column
        tags = line[-1].replace(" & ", "-").split(" ")  # split tag words
        # change a property idenitfier to numeric index
        props = np.array([add_props(prop2idx, idx2prop, prop)
                         for prop in props])
        # change a tag idenitfier to numeric index
        tags = np.array([add_props(tag2idx, idx2tag, tag) for tag in tags])
        # put the numeric and tag ids into image-id-to-prop-tag map
        id2prop[line[0]] = [props, tags]
        idx2name[k] = line[0]  # numeric identifier to image name
        name2idx[line[0]] = k  # image name to numeric index identifier
        name2line[line[0]] = ",".join(line[1:-1])
        # name2rawline[line[0]] = line[0]

    return id2prop, idx2name, name2idx, prop2idx, idx2prop, name2line, tag2idx, idx2tag


def crop_center(img, cropx, cropy):
    """
    Crop a give image in a center with x and y size
    """
    y, x, c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def get_score(id2prop, a, b):

    aprops, atags = id2prop[a]
    bprops, btags = id2prop[b]
    p = np.array_equal(aprops, bprops)
    if p:
        p = 1
    else:
        p = 0
    t = len([x for x in atags if x in btags])
    return(p * t/len(atags))


def find_sims(name, sim_mat, idx2map):
    sims = sim_mat[:, idx2map[name]]
    argsort = np.argsort(sims)
    # print(len(argsort), sim_mat.shape, len(idx2map))
    argsort = np.flip(argsort)[:11]
    return sims[argsort], argsort



def display_similars(name, sim_mat):
    sims, ids = find_sims(name, sim_mat)
    ground_names = [map2idx[x] for x in ids[1:]]
    print(name)

    for idxi in range(len(ids)):
        idx = ids[idxi]
        print(map2idx[idx], sims[idxi], name2line[map2idx[idx]])
    images = [get_image(map2idx[idx]) for idx in ids]

    plt.figure(figsize=(10, 10))  # specifying the overall grid size

    for i in range(len(images)):
        # the number of images in the grid is 5*5 (25)
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i])

    plt.show()


def evaluate(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def get_acc(psim, test, map2idx, idx2map, sim_mat):
    accs = []
    for i in range(len(test)):
        sims, ids = find_sims(test[i], psim, idx2map)
        pred_names = [map2idx[x] for x in ids[1:]]
        sims, ids = find_sims(test[i], sim_mat, idx2map)
        ground_names = [map2idx[x] for x in ids[1:]]
        acc = evaluate(ground_names, pred_names)
        accs.append(acc)

    return np.mean(accs) * 100

def get_unique_classes(df):
    uniques = []
    for col in list( df.columns)[1:-1]:
        uniques.append(list(df[col].unique()))
    return uniques

# functions to encoding string into one-hot encoding

def format_output(outs, props, uniques):
    for i in range(len(props)):
        if i == 6:
            continue
        labels = uniques[i]
        index = labels.index(props[i])
        one_hot = to_onehot(index, len(labels))
        outs[i].append(one_hot)
        
def prepare_outputs(lines, uniques):
    outs = [[], [], [], [],  [], [], [], []]
    for line in lines:
        format_output(outs, line, uniques)
    
    return outs

def to_onehot(val, n_size):
    vec = [0]*n_size
    vec[val] = 1
    return vec

def preproc(line):
    line = line.replace("men's", "men")
    line = line.replace("man", "men")
    line = line.replace("woman", "women")
    line = line.replace("women's", "women")
    line = line.replace(" & ", "-")
    line = line.replace("&", "-")
    line = line.replace("tshirts", "tshirt")
    line = line.replace("t-shirts", "tshirt")
    line = line.replace("'", "")
    return line

def get_tag_data(desc_path):
    lines = open(desc_path).readlines()
    vocab = []
    for k in range(len(lines)):
        line = lines[k]
        line = line.lower().strip().split(",")
        vocab.extend(preproc(line[-1]).split(' '))

    result = Counter(vocab)
    final_vocab = set([])
    for k in result.keys():
        if result[k] > 5:
            final_vocab.add(k)

    start = "<sos>"
    end = "<eos>"
    pad = "<pad>"
    final_vocab.add(start)
    final_vocab.add(end)
    final_vocab.add(pad)

    word2id = {}
    id2word = {}
    vocab = list(final_vocab)
    for i in range(len(final_vocab)):
        size = len(word2id)
        word2id[vocab[i]] = size
        id2word[size] = vocab[i]

    data = []
    max_line = 14
    for k in range(len(lines)):
        line = lines[k]
        line = line.lower().strip().split(",")
        line = preproc(line[-1]).split(" ")
        linex = []
        for li in range(len(line)):
            if line[li] in final_vocab:
                linex.append(line[li])
        linex = list(set(linex))
        linex = [start] + linex + [end]
        left = max_line + 1 - len(linex)
        pads = [pad] * left
        linex = linex + pads
        linex = [word2id[w] for w in linex]
        linex = [[linex[i], linex[i+1]] for i in range(len(linex) - 1)]
        data.append(linex)
        
    data = np.array(data, dtype=np.long)
    tt_data = data[:3000]
    ts_data = data[3000:]
    return tt_data, ts_data, vocab, word2id, id2word


# def get_image_names(df):
#     lines = list(df.iloc[:, 1:-1].itertuples(index=False, name=None))
#     train_names = [str(x) for x in df.iloc[:3000,0]]
#     test_names = [str(x) for x in df.iloc[3000:,0]]
#     return train_names, test_names

# get_tag_data("pca/data/smallx/smallx.csv")