"""Mapping between MANO 21-joint indices and egodex joint names."""

# MANO 21-joint parent indices (for computing bone directions)
MANO_PARENTS = [
    -1,  # 0: wrist (root)
    0,   # 1: thumb CMC
    1,   # 2: thumb MCP
    2,   # 3: thumb IP
    3,   # 4: thumb tip
    0,   # 5: index MCP
    5,   # 6: index PIP
    6,   # 7: index DIP
    7,   # 8: index tip
    0,   # 9: middle MCP
    9,   # 10: middle PIP
    10,  # 11: middle DIP
    11,  # 12: middle tip
    0,   # 13: ring MCP
    13,  # 14: ring PIP
    14,  # 15: ring DIP
    15,  # 16: ring tip
    0,   # 17: little MCP
    17,  # 18: little PIP
    18,  # 19: little DIP
    19,  # 20: little tip
]

# MANO joint index -> egodex joint suffix (side prefix added at runtime)
MANO_TO_EGODEX_SUFFIX = {
    0:  "Hand",
    1:  "ThumbKnuckle",
    2:  "ThumbIntermediateBase",
    3:  "ThumbIntermediateTip",
    4:  "ThumbTip",
    5:  "IndexFingerKnuckle",
    6:  "IndexFingerIntermediateBase",
    7:  "IndexFingerIntermediateTip",
    8:  "IndexFingerTip",
    9:  "MiddleFingerKnuckle",
    10: "MiddleFingerIntermediateBase",
    11: "MiddleFingerIntermediateTip",
    12: "MiddleFingerTip",
    13: "RingFingerKnuckle",
    14: "RingFingerIntermediateBase",
    15: "RingFingerIntermediateTip",
    16: "RingFingerTip",
    17: "LittleFingerKnuckle",
    18: "LittleFingerIntermediateBase",
    19: "LittleFingerIntermediateTip",
    20: "LittleFingerTip",
}

# Egodex joint suffixes that correspond to Metacarpal joints (not in MANO).
# These are interpolated between wrist(0) and the corresponding MCP joint.
METACARPAL_INTERPOLATION = {
    "IndexFingerMetacarpal":  (0, 5),
    "MiddleFingerMetacarpal": (0, 9),
    "RingFingerMetacarpal":   (0, 13),
    "LittleFingerMetacarpal": (0, 17),
}

# All body/non-hand joints in egodex (set to zero confidence)
BODY_JOINTS = [
    "hip",
    "leftShoulder", "leftArm", "leftForearm",
    "rightShoulder", "rightArm", "rightForearm",
    "neck1", "neck2", "neck3", "neck4",
    "spine1", "spine2", "spine3", "spine4", "spine5", "spine6", "spine7",
]


def get_egodex_joint_names():
    """Return all egodex joint names (excluding 'camera') in canonical order."""
    joints = list(BODY_JOINTS)
    for side in ["left", "right"]:
        for idx in sorted(MANO_TO_EGODEX_SUFFIX.keys()):
            suffix = MANO_TO_EGODEX_SUFFIX[idx]
            joints.append(f"{side}{suffix}")
        for suffix in METACARPAL_INTERPOLATION:
            joints.append(f"{side}{suffix}")
    return joints
