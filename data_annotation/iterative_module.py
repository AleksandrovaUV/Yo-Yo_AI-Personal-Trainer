import numpy as np

def bone_length(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# пример: плечо → локоть
def check_upper_arm(keypoints):
    left = bone_length(keypoints[11], keypoints[13])
    right = bone_length(keypoints[12], keypoints[14])
    return left, right
