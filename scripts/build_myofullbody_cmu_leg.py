"""Export the local MyoFullBody + CMU bent-knee bypass as a standalone MJCF."""
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from myowarp.musclemimic.bypass import build
    import mujoco

    spec, model = build()
    output = ROOT / "assets/robots/myofullbody_cmu_leg/myofullbody_cmu_leg.xml"
    tree = ET.fromstring(spec.to_xml())
    size = tree.find("size")
    if size is not None and "memory" in size.attrib:
        size.attrib.pop("njmax", None)
        size.attrib.pop("nconmax", None)
    for parent in tree.iter("default"):
        for default in parent.findall("default"):
            if not default.get("class"):
                default.set("class", "cmu_leg_defaults")
                tree.find(".//body[@name='kneeStator']").set("childclass", "cmu_leg_defaults")
    compiler = tree.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
        # Inertias are already compiled. Reapplying the host bounds would
        # incorrectly clamp the smaller inertias of attached prosthesis parts.
        compiler.attrib.pop("boundmass", None)
        compiler.attrib.pop("boundinertia", None)
    # The assembled spec resolves source assets to absolute paths. Export
    # relative paths so moving/cloning the repository does not break the XML.
    for element in tree.iter():
        file = element.get("file")
        if file and Path(file).is_absolute():
            element.set("file", os.path.relpath(file, output.parent))
        elif file and not (output.parent / file).exists():
            source = ROOT / "assets/robots/myofullbody/model" / file
            if source.is_file():
                element.set("file", os.path.relpath(source, output.parent))
    # Serialize compiled inertials explicitly, including host bodies that
    # obtained their small nonzero masses from compiler bounds.
    for body in tree.iter("body"):
        body_id = model.body(body.get("name")).id
        if model.body_mass[body_id] <= 0:
            continue
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")
        inertial.attrib.clear()
        inertial.set("mass", format(model.body_mass[body_id], ".17g"))
        for name, values in [("pos", model.body_ipos[body_id]),
                             ("quat", model.body_iquat[body_id]),
                             ("diaginertia", model.body_inertia[body_id])]:
            inertial.set(name, " ".join(format(value, ".17g") for value in values))
    output.write_text(ET.tostring(tree, encoding="unicode") + "\n")
    loaded = mujoco.MjModel.from_xml_path(str(output))
    import numpy as np
    for field in ["body_mass", "body_inertia", "geom_size", "geom_contype", "geom_conaffinity"]:
        np.testing.assert_allclose(getattr(loaded, field), getattr(model, field), rtol=1e-5, atol=1e-7)
    data = mujoco.MjData(loaded)
    mujoco.mj_resetDataKeyframe(loaded, data, loaded.key("bypass_stand").id)
    mujoco.mj_forward(loaded, data)
    print(f"Saved {output}\nnq={model.nq}, nv={model.nv}, actuators={model.nu}")
    print(f"Hip-to-prosthetic-knee mount: {loaded.numeric('bypass_hip_to_knee').data[0]:.4f} m")


if __name__ == "__main__":
    main()
