KITCHEN_OBJECT_CLASS_LIST = [
    "Toaster",
    "Spatula",
    "Bread",
    "Mug",
    "CoffeeMachine",
    "Apple",
]

LIVING_ROOM_OBJECT_CLASS_LIST = [
    "Painting",
    "Laptop",
    "Television",
    "RemoteControl",
    "Vase",
    "ArmChair",
]

BEDROOM_OBJECT_CLASS_LIST = ["Blinds", "DeskLamp", "Pillow", "AlarmClock", "CD"]

BATHROOM_OBJECT_CLASS_LIST = ["Mirror", "ToiletPaper", "SoapBar", "Towel", "SprayBottle"]

FULL_OBJECT_CLASS_LIST = (
    KITCHEN_OBJECT_CLASS_LIST
    + LIVING_ROOM_OBJECT_CLASS_LIST
    + BEDROOM_OBJECT_CLASS_LIST
    + BATHROOM_OBJECT_CLASS_LIST
)

MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_UP = "LookUp"
LOOK_DOWN = "LookDown"
DONE = "Done"

DONE_ACTION_INT = 5
GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01