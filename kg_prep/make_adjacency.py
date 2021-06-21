"""
Reads the connectivity graph of each object, and outputs an adjacency matrix to be used in the model.

Returns
adjmat.dat/adjmat_w.dat - the adjacency matrix to be loaded by the model
"""

import argparse
import os
import numpy as np
import json
import scipy.io as sio
import scipy.sparse as sp
import torch

def main(args):
    all_obj = sorted(os.listdir("{}/top_subject_relationships".format(args.data_dir)))
    adjacency = np.zeros((len(all_obj),len(all_obj)), dtype=int)

    for i, obj_file in enumerate(all_obj):
        with open("{}/top_subject_relationships/{}".format(args.data_dir, obj_file), "r") as f:
            data = json.load(f)
        subjects = list(data.keys())
        for obj in subjects:
            if args.weighted:
                adjacency[i][all_obj.index("{}.json".format(obj))] = data[obj]
            else:
                if(data[obj]>=5):
                    adjacency[i][all_obj.index("{}.json".format(obj))] = 1

    if args.weighted:
        torch.save(adjacency, "{}/adjmat_w.dat".format(args.data_dir))
    else:
        torch.save(adjacency, "{}/adjmat.dat".format(args.data_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='kg_prep/kg_data',
                        help="location of kg data directory")
    parser.add_argument('--weighted', default=False, action='store_true',
                        help="compute weighted adjacency matrix")

    args = parser.parse_args()
    main(args)
