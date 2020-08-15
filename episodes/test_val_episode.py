""" Contains the Episodes for Navigation. """
from datasets.environment import Environment
from utils.net_util import gpuify, toFloatTensor
from .basic_episode import BasicEpisode
import pickle
from datasets.data import num_to_name
import json
from datasets.glove import Glove
from utils import flag_parser
from datasets.offline_controller_with_small_rotation import ThorAgentState
import sys

c2p_prob = json.load(open("./data/c2p_prob.json"))
rooms = ['Kitchen', 'Living_Room', 'Bedroom', 'Bathroom']
args = flag_parser.parse_arguments()
metadata_dir = "./data/thor_v1_offline_data/"


import re


class TestValEpisode(BasicEpisode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(TestValEpisode, self).__init__(args, gpu_id, strict_done)
        self.file = None
        self.all_data = None
        self.all_data_enumerator = 0
        self.target_parents = None
        self.room = None

    def _new_episode(self, args, episode):
        """ New navigation episode. """
        scene = episode["scene"]
        if "physics" in scene:
            scene = scene[:-8]

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        self.environment.controller.state = episode["state"]
        self.task_data = episode["task_data"]
        self.target_object = episode["goal_object_type"]
        # with open(metadata_dir + scene + '/visible_object_map.json', 'r')as f :
        #     meta_data = json.load(f)
        #     self.task_data = []
        #     for k, v in meta_data.items():
        #         obj = k.split('|')[0]
        #         if self.target_object == obj:
        #             self.task_data.append(k)
        # regex = re.compile('FloorPlan([0-9]*)')
        # num = int(regex.findall(scene)[0])
        # ind = int(num / 100)
        # if ind > 0:
        #     ind -= 1
        # room = rooms[ind]
        self.room = episode["room"]
        try:
            self.target_parents = c2p_prob[self.room][self.target_object]
        except KeyError:
            print("finding", self.target_object, 'in', self.room)

        if args.verbose:
            print("Scene", scene, "Navigating towards:", self.target_object)
        if args.vis:
            log = open('saved_action_' + args.results_json[:-5] + '.log', "a+")
            sys.stdout = log
            print("Scene", scene, "Navigating towards:", self.target_object)
            print(self.environment.controller.state,self.environment.controller.state.y)
        self.glove_embedding = toFloatTensor(episode["glove_embedding"], self.gpu_id)
        #mark
        #glove = Glove(args.glove_file)
        #self.glove_embedding = toFloatTensor(glove.glove_embeddings[self.target_object], self.gpu_id)
        return True

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        room = None,
        keep_obj=False,
        glove=None,
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None
        self.room = None

        if self.file is None:
            sample_scene = scenes[0]
            if "physics" in sample_scene:
                scene_num = sample_scene[len("FloorPlan") : -len("_physics")]
            else:
                scene_num = sample_scene[len("FloorPlan") :]
            scene_num = int(scene_num)
            scene_type = num_to_name(scene_num)
            task_type = args.test_or_val
            self.file = open(
                "test_val_split/" + scene_type + "_" + task_type + ".pkl", "rb"
            )
            self.all_data = pickle.load(self.file)
            self.file.close()
            self.all_data_enumerator = 0

        episode = self.all_data[self.all_data_enumerator]
        self.all_data_enumerator += 1
        self._new_episode(args, episode)