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

not_my_kps = [1,3,4,6,7,8,9,10,21,22]

def rule_validity(kps):
    """coordiantes validity."""
    issues = []

    for i, kp in enumerate(kps):
        if i not in not_my_kps:
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
    issues = []

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

def rule_proportions(kps):
    """Check limb length ratios (perspective-robust)."""
    issues = []

    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28

    # lengths
    L_upper_arm = dist(kps[L_SH], kps[L_EL])
    L_lower_arm = dist(kps[L_EL], kps[L_WR])
    R_upper_arm = dist(kps[R_SH], kps[R_EL])
    R_lower_arm = dist(kps[R_EL], kps[R_WR])

    L_upper_leg = dist(kps[L_HIP], kps[L_KNEE])
    L_lower_leg = dist(kps[L_KNEE], kps[L_ANK])
    R_upper_leg = dist(kps[R_HIP], kps[R_KNEE])
    R_lower_leg = dist(kps[R_KNEE], kps[R_ANK])

    # helper
    def check_ratio(name, a, b, min_r, max_r):
        if a is None or b is None or a == 0 or b == 0:
            return
        r = a / b
        if r < min_r or r > max_r:
            issues.append({
                "rule": f"{name}_ratio_out_of_range",
                "severity": "WARNING",
                "value": float(r)
            })

    check_ratio("left_arm",  L_upper_arm, L_lower_arm, 0.7, 1.5)
    check_ratio("right_arm", R_upper_arm, R_lower_arm, 0.7, 1.5)

    check_ratio("left_leg",  L_upper_leg, L_lower_leg, 0.8, 1.6)
    check_ratio("right_leg", R_upper_leg, R_lower_leg, 0.8, 1.6)

    return issues


def rule_topology(kps):
    """Check that joints lie between their parent and child."""
    issues = []

    chains = [
        ("left_arm_chain", 11, 13, 15),
        ("right_arm_chain", 12, 14, 16),
        ("left_leg_chain", 23, 25, 27),
        ("right_leg_chain", 24, 26, 28),
    ]

    for name, a, b, c in chains:
        A, B, C = kps[a], kps[b], kps[c]
        if A is None or B is None or C is None:
            continue

        AB = dist(A, B)
        BC = dist(B, C)
        AC = dist(A, C)

        if AB is None or BC is None or AC is None:
            continue

        if AB + BC < AC * 0.7: 
            issues.append({
                "rule": f"{name}_broken_chain",
                "severity": "ERROR",
                "value": float(AB + BC - AC)
            })

    return issues


def rule_forward_bend(kps):
    """Rules for forward bends."""
    issues = []

    NOSE = 0
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    nose = kps[NOSE]
    sh_center = np.mean([kps[L_SH], kps[R_SH]], axis=0) if kps[L_SH] and kps[R_SH] else None
    hip_center = np.mean([kps[L_HIP], kps[R_HIP]], axis=0) if kps[L_HIP] and kps[R_HIP] else None

    if nose is None or sh_center is None or hip_center is None:
        return issues

    spine_len = dist(sh_center, hip_center)
    if spine_len is not None and spine_len > 0.6:  # too long = broken spine
        issues.append({
            "rule": "spine_discontinuity_forward_bend",
            "severity": "WARNING",
            "value": float(spine_len)
        })

    if sh_center[1] > hip_center[1] + 0.15:
        issues.append({
            "rule": "shoulders_below_hips_unexpected",
            "severity": "WARNING",
            "value": float(sh_center[1] - hip_center[1])
        })

    if dist(nose, sh_center) and dist(nose, sh_center) > 0.5:
        issues.append({
            "rule": "head_position_unrealistic_forward_bend",
            "severity": "WARNING"
        })

    return issues




RULES = [rule_validity, rule_anatomy, rule_angles, rule_proportions, rule_topology, rule_forward_bend]

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
