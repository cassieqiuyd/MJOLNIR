"""
Parses through all the subject-relation-object relationships in Visual Genome and counts relationship between a given object from iThor v1.0.1, and its subjects. Outputs the following files/folders:

top_subject_relationships/ - folder containing the connectivity of each object in the KG with its neighbors

all_objects.txt - contains list of all the final objects in the KG
object_relationship_count.json - contains a count of the number of relationships for each object in the KG
relationship_count.json - contains a count of different types of relationships in the KG
"""

import json
import os
import argparse
import difflib
import shutil
from collections import Counter
from misc import *

def main(args):
    print("Loading data...")
    try:
        with open("{}/relationships.json".format(args.data_dir)) as f:
            data = json.load(f)
        print("Done")
    except:
        print("Error loading data. Exiting.")
        exit()
    
    print("Filtering THOR objects from VG...")
    subjects = {}
    for image in data:
        for relationship in image['relationships']:
            subj_name, rel_name, obj_name = get_triplet(relationship)
            
            subj = subjects.get(subj_name, {})
            rel = subj.get(rel_name, {})
            rel[obj_name] = rel.get(obj_name, 0) + 1
            subj[rel_name] = rel
            subjects[subj_name] = subj
    
    # Output data
    with open('{}/subjects_all.json'.format(args.data_dir), 'w') as f:
        json.dump(subjects, f)
    
    subjects_dir = ensuredirs('{}/subjects/'.format(args.data_dir))

    with open('{}/thor_v1_objects.txt'.format(args.data_dir)) as f:
        obj_list = f.readlines()
    
    obj_list_clean = list(map(lambda s: s.strip().lower(), obj_list))
    vg_fulllist = subjects.keys()
    thor_vg_map, vg_list = {}, []

    for i, obj in enumerate(obj_list_clean):
        thor_vg_map[obj_list[i].strip()] = difflib.get_close_matches(obj, possibilities=vg_fulllist, n=5, cutoff=0.9)
        vg_list.extend(thor_vg_map[obj_list[i].strip()])

    with open('{}/thor_vg_map.json'.format(args.data_dir), 'w') as f:
        json.dump(thor_vg_map, f)

    with open('{}/vg_obj.json'.format(args.data_dir), 'w') as f:
        json.dump(list(set(vg_list)), f)
    
    for subj_name, subject in subjects.items():
        if(subj_name not in vg_list):
            continue
        else:
            try:
                with open('{}/{}.json'.format(subjects_dir, subj_name), 'w') as f:
                    json.dump(subject, f)
            # subject name is something wonky so we can't name a file after it /shrug
            except (IOError, UnicodeEncodeError) as e:
                pass
        
    print("Done filtering. Starting object pruning...")
    all_objs = sorted(os.listdir(subjects_dir))

    with open('{}/vg_obj.json'.format(args.data_dir)) as f:
        vg_list = json.load(f)

    with open('{}/thor_vg_map.json'.format(args.data_dir)) as f:
        thor_vg_map = json.load(f)

    bad_thor_objs = ["BathtubBasin", "LaundryHamper", "LaundryHamperLid", "ToiletPaperHanger", "HandTowelHolder"]
    bad_vg_subj = ["binds", "booths", "desser", "ridge", "garbageman", "plater", "plated", "oaster", "swatch"]

    thor_vg_map = {key:val for key, val in thor_vg_map.items() if key not in bad_thor_objs}
    
    for key, value in thor_vg_map.items():
        thor_vg_map[key] = [x for x in value if x not in bad_vg_subj]

    with open('{}/thor_vg_map.json'.format(args.data_dir), 'w') as f:
        json.dump(thor_vg_map, f)

    vg_list = [x for x in vg_list if x not in bad_vg_subj]

    with open('{}/vg_obj.json'.format(args.data_dir), 'w') as f:
        json.dump(vg_list, f)

    for obj in all_objs:
        with open('{}/{}'.format(subjects_dir, obj)) as f:
            data = json.load(f)

        for rel, rel_objs in data.items():
            rel_objs = dict((k, v) for (k, v) in rel_objs.items() if k in vg_list)
            data[rel] = rel_objs
        data = {k: v for k, v in data.items() if len(v)>0}
        if(len(data) == 0):
            continue
        output_noisy_path = ensuredirs(os.path.join("{}/{}/".format(args.data_dir,"final_noisy_rels")))
        with open('{}/{}'.format(output_noisy_path, obj), 'w') as f:
            json.dump(data, f)
    
    all_noisy_objs = sorted(os.listdir(output_noisy_path))
    all_noisy_objs = [s.split('.')[0] for s in all_noisy_objs]

    for thor, vg in thor_vg_map.items():
        final_data = {}
        for vg_obj in vg:
            if(vg_obj not in all_noisy_objs):
                continue
            else:
                with open('{}/{}.json'.format(output_noisy_path, vg_obj)) as f:
                    data = json.load(f)
                
                if(len(final_data) == 0):
                    final_data = data
                else:
                    for rel, rel_objs in data.items():
                        final_data = add_or_append(final_data, rel, rel_objs)
        
        output_path = ensuredirs(os.path.join("{}/{}/".format(args.data_dir,"final_rels")))
        with open('{}/{}.json'.format(output_path, thor), 'w') as f:
            json.dump(final_data, f)

    all_noisy_subjs = sorted(os.listdir(output_path))

    for thor_obj in all_noisy_subjs:
        final_data = {}
        with open('{}/{}'.format(output_path, thor_obj)) as f:
            data = json.load(f)
        
        for rel, rel_objs in data.items():
            final_data[rel] = {}
            subj_list = rel_objs.keys()

            for subj in subj_list:
                thor_key = [list(thor_vg_map.keys())[list(thor_vg_map.values()).index(vg_objs)] for ind, vg_objs in enumerate(list(thor_vg_map.values())) if subj in vg_objs][0]
                
                if(thor_key not in final_data[rel].keys()):
                    final_data[rel][thor_key] = data[rel][subj]
                else:
                    final_data[rel][thor_key] += data[rel][subj]

        final_output_path = ensuredirs(os.path.join("{}/{}/".format(args.data_dir,"refined_final_objs")))

        with open('{}/{}'.format(final_output_path, thor_obj), 'w') as f:
            json.dump(final_data, f)

    print("Done object pruning. Starting relationship pruning...")
    with open('{}/relationship_alias.txt'.format(args.data_dir)) as f:
        rel_list = f.readlines()
    
    rel_list = [s.strip('\n') for s in rel_list]
    
    rel_alias_map = {}
    
    for rel in rel_list:
        rel_key = rel.split(',')[0]

        if(rel_key not in rel_alias_map.keys()):
            rel_alias_map[rel_key] = list(rel.split(','))
        else:
            rel_alias_map[rel_key] = list(set(rel_alias_map[rel_key] + list(rel.split(','))))
    
    all_objs = os.listdir(os.path.join(args.data_dir, "refined_final_objs"))
    
    for obj_file in all_objs:
        final_data = {}
        with open('{}/refined_final_objs/{}'.format(args.data_dir, obj_file)) as f:
            data = json.load(f)

        for rel, subj_dict in data.items():
            if(rel == ""):
                continue
            else:
                rel = rel.replace("  ", " ")

            try:
                rel_key = [list(rel_alias_map.keys())[list(rel_alias_map.values()).index(subj_rels)] for ind, subj_rels in enumerate(list(rel_alias_map.values())) if rel in subj_rels][0]
            except (IndexError) as e:
                final_data[rel] = subj_dict

            if(rel_key not in final_data.keys()):
                final_data[rel_key] = subj_dict
            else:
                for subj in subj_dict.keys():
                    if(subj not in final_data[rel_key].keys()):
                        final_data[rel_key][subj] = subj_dict[subj]
                    else:
                        final_data[rel_key][subj] += subj_dict[subj]

        final_output_path = ensuredirs(os.path.join("{}/{}/".format(args.data_dir,"refined_final_rels")))

        with open('{}/{}'.format(final_output_path, obj_file), 'w') as f:
            json.dump(final_data, f)

    all_objs = os.listdir(os.path.join(args.data_dir, "refined_final_rels"))
    all_objs_list = sorted([s.split('.')[0] for s in all_objs])
    all_rels_list = []
    obj_rel_list = []

    for obj_file in all_objs:
        with open('{}/refined_final_rels/{}'.format(args.data_dir, obj_file)) as f:
            data = json.load(f)
        
        for rel, _ in data.items():
            obj_rel_list.append(obj_file.split('.')[0])
            all_rels_list.append(rel)

    obj_rel_count_dict = Counter(obj_rel_list)
    obj_rel_count_dict = {k: v for k, v in sorted(obj_rel_count_dict.items(), key=lambda item: item[1], reverse = True)}

    rel_count_dict = Counter(all_rels_list)
    rel_count_dict = {k: v for k, v in sorted(rel_count_dict.items(), key=lambda item: item[1], reverse = True)}

    all_rels_list = sorted(list(set(all_rels_list)))
    # print(len(all_rels_list), all_rels_list)

    with open('{}/all_objects.txt'.format(args.data_dir), 'w') as f:
        for item in all_objs_list:
            f.write("%s\n" % item)

    with open('{}/all_relationships.txt'.format(args.data_dir), 'w') as f:
        for item in all_rels_list:
            f.write("%s\n" % item)

    with open('{}/relationship_count.json'.format(args.data_dir), 'w') as f:
        json.dump(rel_count_dict, f, indent=1)

    with open('{}/object_relationship_count.json'.format(args.data_dir), 'w') as f:
        json.dump(obj_rel_count_dict, f, indent=1)

    print("Done relationship pruning. Saving top relationships...")
    all_objs = os.listdir(os.path.join(args.data_dir, "refined_final_rels"))
    all_objs = sorted([s.split('.')[0] for s in all_objs])

    for obj in all_objs:
        obj_rel_dict = {}
        with open('{}/refined_final_rels/{}.json'.format(args.data_dir, obj)) as f:
            data = json.load(f)
        
        for rel, subj_dict in data.items():
            if(obj_rel_dict is None):
                obj_rel_dict = subj_dict
            else:
                for subj in subj_dict.keys():
                    if(subj not in obj_rel_dict.keys()):
                        obj_rel_dict[subj] = subj_dict[subj]
                    else:
                        obj_rel_dict[subj] += subj_dict[subj]

        obj_rel_dict = {k: v for k, v in sorted(obj_rel_dict.items(), key=lambda item: item[1], reverse = True)}

        final_output_path = ensuredirs(os.path.join("{}/{}/".format(args.data_dir,"top_subject_relationships")))

        with open('{}/{}.json'.format(final_output_path, obj), 'w') as f:
            json.dump(obj_rel_dict, f, indent=1)
        
    print("Removing extra files and folders...")
    shutil.rmtree("{}/final_noisy_rels/".format(args.data_dir))
    shutil.rmtree("{}/final_rels/".format(args.data_dir))
    shutil.rmtree("{}/refined_final_objs/".format(args.data_dir))
    shutil.rmtree("{}/refined_final_rels/".format(args.data_dir))
    shutil.rmtree("{}/subjects/".format(args.data_dir))
    
    os.remove("{}/vg_obj.json".format(args.data_dir))
    os.remove("{}/thor_vg_map.json".format(args.data_dir))
    os.remove("{}/subjects_all.json".format(args.data_dir))
    os.remove("{}/all_relationships.txt".format(args.data_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='kg_prep/kg_data',
                        help="location of kg data directory")

    args = parser.parse_args()
    main(args)
