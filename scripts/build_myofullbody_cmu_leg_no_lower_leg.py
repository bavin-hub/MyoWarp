"""Export a separate CMU-leg model with biological right knee/distal bodies removed."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/robots/myofullbody_cmu_leg/myofullbody_cmu_leg.xml"
OUTPUT = (
    ROOT
    / "assets/robots/myofullbody_cmu_leg_no_lower_leg/myofullbody_cmu_leg_no_lower_leg.xml"
)


def main():
    original = mujoco.MjModel.from_xml_path(str(SOURCE))
    tree = ET.parse(SOURCE).getroot()
    femur = tree.find(".//body[@name='femur_r']")
    removed = {
        key: set() for key in ("body", "joint", "geom", "site", "tendon", "actuator")
    }
    for name in ("tibia_r", "patella_r"):
        body = femur.find(f"body[@name='{name}']")
        for element in body.iter():
            if element.tag in removed and element.get("name"):
                removed[element.tag].add(element.get("name"))
        femur.remove(body)
    tendons = tree.find("tendon")
    for tendon in list(tendons):
        if any(
            child.get(key) in removed[key]
            for child in tendon.iter()
            for key in ("site", "geom", "joint")
        ):
            removed["tendon"].add(tendon.get("name"))
            tendons.remove(tendon)
    actuators = tree.find("actuator")
    for actuator in list(actuators):
        if (
            actuator.get("tendon") in removed["tendon"]
            or actuator.get("joint") in removed["joint"]
        ):
            removed["actuator"].add(actuator.get("name"))
            actuators.remove(actuator)
    refs = {
        "body": "body",
        "body1": "body",
        "body2": "body",
        "joint": "joint",
        "joint1": "joint",
        "joint2": "joint",
        "site": "site",
        "site1": "site",
        "site2": "site",
        "geom": "geom",
        "geom1": "geom",
        "geom2": "geom",
        "tendon": "tendon",
        "tendon1": "tendon",
        "tendon2": "tendon",
        "actuator": "actuator",
    }
    for section in ("equality", "contact", "sensor"):
        parent = tree.find(section)
        if parent is None:
            continue
        for element in list(parent):
            invalid = any(
                element.get(attr) in removed[kind] for attr, kind in refs.items()
            )
            for prefix in ("obj", "ref"):
                kind = element.get(prefix + "type")
                if kind in removed and element.get(prefix + "name") in removed[kind]:
                    invalid = True
            if invalid:
                parent.remove(element)
    # The serialized key is mapped by surviving joint names, never truncated.
    keys = tree.find("keyframe")
    if keys is not None:
        tree.remove(keys)
    for element in tree.iter():
        file = element.get("file")
        if file:
            element.set(
                "file", os.path.relpath((SOURCE.parent / file).resolve(), OUTPUT.parent)
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(ET.tostring(tree, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    pose = model.qpos0.copy()
    old_pose = original.key("bypass_stand").qpos
    for j in range(model.njnt):
        src = original.joint(model.joint(j).name)
        width = {0: 7, 1: 4}.get(int(model.jnt_type[j]), 1)
        pose[model.jnt_qposadr[j] : model.jnt_qposadr[j] + width] = old_pose[
            src.qposadr[0] : src.qposadr[0] + width
        ]
    keys = ET.SubElement(tree, "keyframe")
    ET.SubElement(
        keys,
        "key",
        name="bypass_stand",
        qpos=" ".join(format(float(q), ".17g") for q in pose),
    )
    OUTPUT.write_text(ET.tostring(tree, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    for i in range(model.nbody):
        name = model.body(i).name
        if not name:
            continue
        source = original.body(name).id
        np.testing.assert_allclose(
            model.body_mass[i], original.body_mass[source], rtol=1e-10
        )
        np.testing.assert_allclose(
            model.body_inertia[i], original.body_inertia[source], rtol=1e-10
        )
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "nq": model.nq,
        "nv": model.nv,
        "na": model.na,
        "nu": model.nu,
        "source_mass_kg": float(original.body_mass.sum()),
        "mass_kg": float(model.body_mass.sum()),
        "removed": {k: sorted(v) for k, v in removed.items()},
    }
    (OUTPUT.parent / "anatomy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "removed"}, indent=2))
    print("Removed muscle actuators:", len(removed["actuator"]))


if __name__ == "__main__":
    main()
