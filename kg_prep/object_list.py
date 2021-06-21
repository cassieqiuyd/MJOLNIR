"""
Runs the AI2THOR controller, and stores a list of all the objects present.

Returns
thor_v1_objects.txt - the file containing the list of all objects in iThor
"""

from ai2thor import controller
controller = controller.Controller()
controller.start()

rooms = [['Kitchen',0],['Living_Room',200],['Bedroom',300],['Bathroom',400]]
obj_ls = []
for k in range(len(rooms)):#len(rooms)
    room = rooms[k][0]
    room_ind = rooms[k][1]
    for i in range(30):    #for each scene
        controller.reset('FloorPlan'+str(1+i+room_ind))
        event = controller.step(dict(action='Initialize'))
        n = len(event.metadata["objects"])
        lst = []
        for j in range(n):   #for each object
            obj = event.metadata["objects"][j]['objectType']
            if obj not in obj_ls:
                obj_ls.append(obj)

f= open("kg_prep/kg_data/thor_v1_objects.txt","w+")
for i in range(len(obj_ls)-1):
    f.write(sorted(obj_ls)[i]+'\n')
f.write(sorted(obj_ls)[-1])
f.close()