import os
import os.path as osp

'''
Deal with some idiosyncrasies in Visual Genome, lowercase everything.
Returns subject, relation, object strings.
'''
def get_triplet(relationship):
    try:
        subj_name = relationship['subject']['names'][0]
    except KeyError:
        subj_name = relationship['subject']['name']
    try:
        obj_name = relationship['object']['names'][0]
    except:
        obj_name = relationship['object']['name']
    return subj_name.lower(), relationship['predicate'].lower(), obj_name.lower()

def add_or_append(final_dict, rel, rel_objs):
    if rel not in final_dict.keys():
        final_dict[rel] = rel_objs
        return final_dict
    else:
        for subj, count in final_dict[rel].items():
            if(subj in rel_objs.keys()):
                count+=rel_objs[subj]
            final_dict[rel][subj] = count
    return final_dict

def ensuredirs(fpath):
    fdir = osp.dirname(fpath)
    if not osp.exists(fdir): os.makedirs(fdir)
    return fpath