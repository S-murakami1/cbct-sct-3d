"""Map TotalSegmentator `total` (117 classes) to the AFP paper's 7 labels.

Longuefosse et al., arXiv:2410.10328 — fused TotalSeg V2 groups:
  0 background, 1 organs, 2 cardiac, 3 muscles, 4 bones, 5 ribs, 6 vertebrae

Bones are peeled from TotalSegmentator's muscles 5-part (skull, hip, ...).
Brain and spinal_cord sit in that same 5-part for packing only; they are
mapped to organs, not muscles.
"""
from __future__ import annotations

from totalsegmentator.map_to_binary import class_map, class_map_5_parts

NEW_LABELS: dict[int, str] = {
    0: "background",
    1: "organs",
    2: "cardiac",
    3: "muscles",
    4: "bones",
    5: "ribs",
    6: "vertebrae",
}

NEW_LABEL_IDS: dict[str, int] = {v: k for k, v in NEW_LABELS.items()}

ORGAN_FROM_MUSCLES_PART = {
    "brain",
    "spinal_cord",
}

BONE_NAMES = {
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "skull",
}

SEVEN_COLORS = {
    0: (0.00, 0.00, 0.00),
    1: (0.90, 0.62, 0.00),  # organs — orange
    2: (0.84, 0.37, 0.00),  # cardiac — vermillion
    3: (0.80, 0.47, 0.65),  # muscles — pink
    4: (0.94, 0.89, 0.26),  # bones — yellow
    5: (0.34, 0.71, 0.91),  # ribs — sky
    6: (0.00, 0.62, 0.45),  # vertebrae — green
}


def _names(part: str) -> set[str]:
    return set(class_map_5_parts[part].values())


def group_names() -> dict[int, set[str]]:
    organs = _names("class_map_part_organs")
    cardiac = _names("class_map_part_cardiac")
    ribs = _names("class_map_part_ribs")
    vertebrae = _names("class_map_part_vertebrae")
    muscles_all = _names("class_map_part_muscles")
    bones = BONE_NAMES
    organ_extra = ORGAN_FROM_MUSCLES_PART
    missing_bones = bones - muscles_all
    missing_organs = organ_extra - muscles_all
    if missing_bones:
        raise RuntimeError(f"bone names not in muscles part: {sorted(missing_bones)}")
    if missing_organs:
        raise RuntimeError(f"organ names not in muscles part: {sorted(missing_organs)}")
    muscles = muscles_all - bones - organ_extra
    organs = organs | organ_extra
    return {
        1: organs,
        2: cardiac,
        3: muscles,
        4: bones,
        5: ribs,
        6: vertebrae,
    }


def correspondence_rows() -> list[dict[str, object]]:
    total = class_map["total"]
    name_to_new: dict[str, int] = {}
    for new_id, names in group_names().items():
        for name in names:
            if name in name_to_new:
                raise RuntimeError(f"name {name} mapped twice")
            name_to_new[name] = new_id

    rows: list[dict[str, object]] = [
        {
            "old_id": 0,
            "old_name": "background",
            "new_id": 0,
            "new_name": "background",
        }
    ]
    unmapped: list[str] = []
    for old_id in sorted(total):
        old_name = total[old_id]
        new_id = name_to_new.get(old_name)
        if new_id is None:
            unmapped.append(f"{old_id}:{old_name}")
            continue
        rows.append(
            {
                "old_id": int(old_id),
                "old_name": old_name,
                "new_id": int(new_id),
                "new_name": NEW_LABELS[new_id],
            }
        )
    if unmapped:
        raise RuntimeError(f"unmapped total classes: {unmapped}")
    return rows


def old_to_new_lut(n: int = 256) -> list[int]:
    lut = [0] * n
    for row in correspondence_rows():
        lut[int(row["old_id"])] = int(row["new_id"])
    return lut


def correspondence_payload() -> dict:
    rows = correspondence_rows()
    reverse: dict[str, list[dict[str, object]]] = {name: [] for name in NEW_LABELS.values()}
    for row in rows:
        reverse[str(row["new_name"])].append(
            {"old_id": row["old_id"], "old_name": row["old_name"]}
        )
    return {
        "scheme": "AFP TotalSeg fused 7 labels (Longuefosse arXiv:2410.10328)",
        "source_task": "total",
        "new_labels": {str(k): v for k, v in NEW_LABELS.items()},
        "old_to_new": rows,
        "new_to_old": reverse,
    }
