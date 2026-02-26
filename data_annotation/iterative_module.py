

import numpy as np
import math


def dist(a, b):
    if a is None or b is None:
        return None
    return np.linalg.norm(np.array(a) - np.array(b))

def angle(a, b, c):

    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return None
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def rule_validity(kps):
    """coordiantes validity."""
    issues = []

    for i, kp in enumerate(kps):
        if kp is None:
            issues.append({"rule": f"kp_{i}_missing", "severity": "ERROR"})
            continue
        x, y = kp
        if not (0 <= x <= 1 and 0 <= y <= 1):
            issues.append({"rule": f"kp_{i}_out_of_bounds", "severity": "ERROR"})

    if sum(k is None for k in kps) > len(kps) * 0.3:
        issues.append({"rule": "too_many_missing_keypoints", "severity": "ERROR"})

    return issues


def rule_anatomy(kps):
    """anatomic rules."""
    issues = []

    # MediaPipe
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28


    limbs = [
        ("left_upper_arm",  L_SH, L_EL),
        ("left_lower_arm",  L_EL, L_WR),
        ("right_upper_arm", R_SH, R_EL),
        ("right_lower_arm", R_EL, R_WR),
        ("left_upper_leg",  L_HIP, L_KNEE),
        ("left_lower_leg",  L_KNEE, L_ANK),
        ("right_upper_leg", R_HIP, R_KNEE),
        ("right_lower_leg", R_KNEE, R_ANK),
    ]

    lengths = []
    for name, a, b in limbs:
        d = dist(kps[a], kps[b])
        if d is not None:
            lengths.append(d)

    if len(lengths) < 4:
        return issues

    mean_len = np.mean(lengths)

    for name, a, b in limbs:
        d = dist(kps[a], kps[b])
        if d is None:
            continue
        if d > mean_len * 2.5:
            issues.append({"rule": f"{name}_too_long", "severity": "WARNING", "value": float(d)})
        if d < mean_len * 0.2:
            issues.append({"rule": f"{name}_too_short", "severity": "WARNING", "value": float(d)})

    return issues




def rule_angles(kps):
    """angle rules."""
    issues = []

    joints = [
        ("left_elbow", 11, 13, 15),
        ("right_elbow", 12, 14, 16),
        ("left_knee", 23, 25, 27),
        ("right_knee", 24, 26, 28),
    ]

    for name, a, b, c in joints:
        ang = angle(kps[a], kps[b], kps[c])
        if ang is None:
            continue
        if ang < 5 or ang > 185:
            issues.append({"rule": f"{name}_angle_out_of_range", "severity": "WARNING", "value": float(ang)})

    return issues



RULES = [rule_validity, rule_anatomy, rule_angles]

def check_pose(kps):
    all_issues = []

    for rule in RULES:
        issues = rule(kps)
        if issues:
            all_issues.extend(issues)

    if any(i["severity"] == "ERROR" for i in all_issues):
        status = "ERROR"
    elif any(i["severity"] == "WARNING" for i in all_issues):
        status = "WARNING"
    else:
        status = "OK"

    return {
        "status": status,
        "issues": all_issues
    }
