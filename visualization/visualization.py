import ai2thor.controller
import numpy as np
import argparse
import cv2
import time
import json

def data_loader(obj_file, action):
    # load object list
    obj_list = []
    with open(obj_file) as f:
        line = f.readlines()
        for l in line:
            obj_list.append(l.strip())

    # parse actions in each episode
    total_list = []
    success_list = []
    i = -1
    with open(action, 'r') as f:
        for line in f.readlines():
            if line.startswith('Scene'):
                new_list = []
                FloorPlan = line.split()[1]
                Obj = line.split()[-1]
                new_list.append(FloorPlan)
                new_list.append(Obj)
            elif '|' in line:
                y = line.split()[-1]
                x = line.split()[0].split('|')
                x.insert(1, y)
                new_list.append(x)
            elif line.startswith('{'):
                new_list.append(line.split(':')[1][2:-3])
            if line.startswith("Success"):
                new_list.insert(3, line.split()[1])
                if line.endswith("True\n"):
                    success_list.append(i)
                continue
            if line.startswith('tensor'):
                continue
            if new_list[-1] == 'Done':
                total_list.append(new_list)
                i += 1
    return obj_list, total_list


def target_parent(ep1,c2p):
    fp = ep1[0]
    scene_num = int(fp[9:])
    scene_type = ['Kitchen', '','Living_Room', 'Bedroom', 'Bathroom'][int(scene_num / 100)]
    print(fp)
    target_parents = c2p[scene_type][ep1[1]]
    return target_parents


def img_bbx(args, event, action, obj_list, ep1, target_parents):
    """
    generate image with object bounding box based on ai2thor
    :param args:
    :param event: ai2thor event, room visualization
    :param action: current action
    :param obj_list: object list
    :param ep1: episode
    :param target_parents: objects' top3 parent
    :return: images
    """

    blue = (253, 152, 0)
    white = (255, 255, 255)  # white
    red = (0,45,255)
    green = (25, 78,55)
    black = (0, 0, 0)
    thickness = 1
    font_scale = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = cv2.UMat(event.cv2img)
    vis_dict = {}
    for i in range(len(event.metadata['objects'])):
        vis_dict[event.metadata['objects'][i]['objectType']] = event.metadata['objects'][i]['visible']
    for obj in event.class_detections2D:
        if obj not in obj_list:
            continue
        lx = event.class_detections2D[obj][0][2] - event.class_detections2D[obj][0][0]
        ly = event.class_detections2D[obj][0][3] - event.class_detections2D[obj][0][1]
        bbx = lx * ly
        if bbx < 200:
            continue
        scale = lx / 300 * 5
        scale = np.clip(scale, 0, 0.7)
        x1, y1, x2, y2 = event.class_detections2D[obj][0]

        if obj in target_parents:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), red, 1)
            img = cv2.putText(img, obj, (x1, y2-2), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        elif obj == ep1[1]:
            img = cv2.rectangle(img,(x1,y1),(x2, y2),blue, 2)
            img = cv2.putText(img, obj, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
        elif args.show_all_obj:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), white, 1)
            img = cv2.putText(img, obj, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, white, 1)


    text = action
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=1.5, thickness=2)[0]
    box_coords = ((0, 0), (192, 30))
    img = cv2.rectangle(img, box_coords[0], box_coords[1], white, cv2.FILLED)
    img = cv2.putText(img, action, (10, 25), font, 1, black, 2)

    img2 = 255*np.ones((500,500,3), np.uint8)
    img2 = cv2.UMat(img2)
    img2 = cv2.putText(img2, 'Goal: '+ep1[1], (20, 40), font, 1, blue, 2)
    img2 = cv2.putText(img2, 'Parent: ', (20, 100), font, 1, red, 2)
    for tp_i, tp in enumerate(target_parents):
        img2 = cv2.putText(img2, tp, (20, 140+tp_i*30), font, 1, red, 2)

    img2 = cv2.putText(img2, 'Success: '+ ep1[3], (20, 280), font, 1, black, 2)

    img_array = cv2.UMat.get(img)
    img_array2 = cv2.UMat.get(img2)
    img = np.hstack((img_array,img_array2))

    winname = "Test"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 510, 250)
    cv2.imshow(winname, img)

    cv2.waitKey(2)

    return img

def start_controller(args, ep1, obj_list, target_parents):
    controller = ai2thor.controller.Controller()
    controller.start(player_screen_height=500, player_screen_width=500)
    controller.reset(ep1[0])
    event = controller.step(dict(action='Initialize', gridSize=0.25, fieldOfView=90, renderObjectImage=True))
    event = controller.step(dict(action='TeleportFull', x=ep1[2][0], y=ep1[2][1], z=ep1[2][2],
                                 rotation=ep1[2][3], horizon=ep1[2][4]))
    frames = []
    rotation_list = ['RotateRight', 'RotateLeft']
    angle = 0
    for i in range(len(ep1)-5):
        print(ep1[i+4])
        time.sleep(0.5)
        img = img_bbx(args, event, ep1[i+4], obj_list, ep1, target_parents)
        pos = event.metadata['agent']['position']
        rot = event.metadata['agent']['rotation']
        if ep1[i+4] == 'RotateLeft':
            event = controller.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=pos['z'],
                                 rotation=rot['y']-45, horizon = angle))
        elif ep1[i+4] == 'RotateRight':
            event = controller.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=pos['z'],
                                 rotation=rot['y']+45, horizon = angle))
        elif ep1[i+4] == 'LookDown':
            event = controller.step(dict(action=ep1[i+4]))
            angle += 30
            angle = np.clip(angle,-60,60)
        elif ep1[i+4] == 'LookUp':
            event = controller.step(dict(action=ep1[i+4]))
            angle -= 30
            angle = np.clip(angle,-60,60)
        else:
            event = controller.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=pos['z'],
                                  rotation=rot['y'], horizon = angle))
            event = controller.step(dict(action=ep1[i+4]))
        frames.append(img)
        time.sleep(1.5)
    img = img_bbx(args, event, ep1[-1], obj_list, ep1, target_parents)
    frames.append(img)
    time.sleep(3)
    controller.stop()
    cv2.destroyAllWindows()

    return frames

def export_video(frames):
    height, width, layers=frames[1].shape
    print("saving video with resolution %s, %s", width, height)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter('mjolnir.mp4',fourcc,1.2,(width,height))

    for j in frames:
        video.write(j)
    video.release()

if __name__ == '__main__':
    if ai2thor.__version__ != "1.0.1":
        print("The current version is:", ai2thor.__version__)
        print("The preferred version is 1.0.1")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--episode', type=int, default=-7, help="episode to visualize.")
    parser.add_argument('--actionList', type=str, default="action_obj_r.log", help="action log.")
    parser.add_argument('--export_video', default=False, action='store_true')
    parser.add_argument('--show_all_obj', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    obj_file = 'all_objects_v1.txt'
    obj_list, total_list = data_loader(obj_file, args.actionList)
    c2p = json.load(open('c2p_top3.json','r'))

    episode = total_list[args.episode]
    target_parents = target_parent(episode, c2p)
    frames = start_controller(args, episode, obj_list, target_parents)


    if args.export_video:
        export_video(frames)
