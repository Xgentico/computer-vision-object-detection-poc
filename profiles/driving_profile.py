# =========================
# DRIVING DETECTION PROFILE
# =========================

DETECTION_PROFILE = "driving"

# YOLO COCO class IDs for the driving scenario
#
# 0  = person
# 1  = bicycle
# 2  = car
# 3  = motorcycle
# 5  = bus
# 7  = truck
# 9  = traffic light
# 11 = stop sign
TARGET_CLASS_IDS = [0, 1, 2, 3, 5, 7, 9, 11]

# Person class is counted uniquely
PERSON_CLASS_ID = 0