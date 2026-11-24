"""Build H1-2 + CMU leg with a distal adapter so both feet meet the ground."""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "assets/robots/unitree_h1_2/xmls/h1_2.xml"
    cmu_source = ROOT / "assets/robots/cmu_leg/prosthesis.xml"
    output = ROOT / "assets/robots/h1_with_cmu_leg/h1_with_cmu_leg.xml"
    tree, cmu = ET.parse(source), ET.parse(cmu_source).getroot()
    model = tree.getroot()
    model.set("model", "h1_with_cmu_leg")
    model.find("compiler").set("meshdir", ".")
    for mesh in model.findall("asset/mesh"):
        mesh.set("file", "../unitree_h1_2/xmls/assets/" + mesh.get("file"))
    for mesh in cmu.findall("asset/mesh"):
        mesh = deepcopy(mesh)
        mesh.set("file", "../cmu_leg/meshes/" + mesh.get("file"))
        model.find("asset").append(mesh)
    defaults = deepcopy(cmu.find("default"))
    defaults.set("class", "cmu_leg")
    defaults.find("default").set("class", "cmu_collision")
    model.find("default").append(defaults)
    thigh = model.find(".//body[@name='right_hip_roll_link']")
    original_knee = thigh.find("body[@name='right_knee_link']")
    leg = deepcopy(cmu.find("worldbody/body[@name='kneeStator']"))
    leg.set("pos", original_knee.get("pos"))
    # Source +Y up, +X forward; H1 +Z up, +X forward. Knee -Z -> H1 +Y.
    leg.set("quat", "0.7071067811865476 0.7071067811865476 0 0")
    leg.set("childclass", "cmu_leg")
    for geom in leg.iter("geom"):
        if geom.get("class") == "collision":
            geom.set("class", "cmu_collision")
    # The source foot mesh stops 13 mm above its contact spheres. Render a
    # lightweight outsole spanning those contacts, without adding collisions,
    # mass, or moving the physical foot below the ground.
    foot = leg.find(".//body[@name='ankle_foot']")
    ET.SubElement(
        foot,
        "geom",
        name="cmu_visible_outsole",
        type="box",
        pos="0.033 -0.1015 0.00187",
        size="0.135 0.0065 0.035",
        contype="0",
        conaffinity="0",
        density="0",
        group="2",
        rgba="0.15 0.15 0.15 1",
    )
    thigh.remove(original_knee)
    thigh.append(leg)
    for section in ("equality", "tendon", "actuator", "sensor"):
        target = model.find(section)
        if target is None:
            target = ET.SubElement(model, section)
        for element in cmu.find(section):
            target.append(deepcopy(element))
    world = model.find("worldbody")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="3 3 0.1", rgba="0.7 0.7 0.7 1"
    )
    ET.SubElement(world, "light", pos="0 -2 3", dir="0 0 -1", directional="true")
    ET.SubElement(
        model,
        "option",
        timestep="0.001",
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    # Fit mounting height in the neutral pose; preserve all source link lengths.
    compiled = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    left = compiled.geom("left_foot4_collision").id
    left_bottom = data.geom_xpos[left, 2] - compiled.geom_size[left, 0]
    cmu_bottom = min(
        data.geom_xpos[compiled.geom(n).id, 2]
        - compiled.geom_size[compiled.geom(n).id, 0]
        for n in ("heel", "toe")
    )
    extension = cmu_bottom - left_bottom
    old_mount = np.fromstring(leg.get("pos"), sep=" ")
    thigh_rotation = data.xmat[compiled.body("right_hip_roll_link").id].reshape(3, 3)
    new_mount = old_mount + thigh_rotation.T @ np.array([0.0, 0.0, -extension])
    leg.set("pos", " ".join(map(str, new_mount)))
    # Connect the old H1 knee location to the top of the CMU socket housing.
    socket = new_mount + np.array([0.0, 0.0, 0.0775])
    ET.SubElement(
        thigh,
        "geom",
        name="cmu_mount_adapter",
        type="capsule",
        **{
            "class": "visual",
            "size": "0.027",
            "fromto": " ".join(map(str, np.r_[old_mount, socket])),
            "rgba": "0.55 0.55 0.55 1",
        },
    )
    pelvis = model.find("worldbody/body[@name='pelvis']")
    pelvis_pos = np.fromstring(pelvis.get("pos"), sep=" ")
    pelvis_pos[2] -= left_bottom
    pelvis.set("pos", " ".join(map(str, pelvis_pos)))
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    print(
        f"{output}\nCMU knee mount lowered {extension * 100:.2f} cm; both soles on ground."
    )


if __name__ == "__main__":
    main()
