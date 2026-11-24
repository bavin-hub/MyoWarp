# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0; see LICENSE and NOTICE in this directory.
# Ported from MyoSuite 3.0.0; imports adapted for standalone MyoWarp use.

"""Repository-local MyoFullBody builder, matching the checkpoint model release."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import mujoco
from .model_versions import apply_models_version

FINGER_JOINT_TOKENS = (
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "md2_flexion_r",
    "md3_flexion_r",
    "md4_flexion_r",
    "md5_flexion_r",
    "pm2_flexion_r",
    "pm3_flexion_r",
    "pm4_flexion_r",
    "pm5_flexion_r",
    "cmc_flexion_l",
    "cmc_abduction_l",
    "mp_flexion_l",
    "ip_flexion_l",
    "mcp2_flexion_l",
    "mcp2_abduction_l",
    "mcp3_flexion_l",
    "mcp3_abduction_l",
    "mcp4_flexion_l",
    "mcp4_abduction_l",
    "mcp5_flexion_l",
    "mcp5_abduction_l",
    "md2_flexion_l",
    "md3_flexion_l",
    "md4_flexion_l",
    "md5_flexion_l",
    "pm2_flexion_l",
    "pm3_flexion_l",
    "pm4_flexion_l",
    "pm5_flexion_l",
)


FINGER_MUSCLE_TOKENS = (
    "FDS2",
    "FDS3",
    "FDS4",
    "FDS5",
    "FDP2",
    "FDP3",
    "FDP4",
    "FDP5",
    "EDC2",
    "EDC3",
    "EDC4",
    "EDC5",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "RI3",
    "RI4",
    "RI5",
    "LU_RB2",
    "LU_RB3",
    "LU_RB4",
    "LU_RB5",
    "UI_UB2",
    "UI_UB3",
    "UI_UB4",
    "UI_UB5",
    "FDS2_left",
    "FDS3_left",
    "FDS4_left",
    "FDS5_left",
    "FDP2_left",
    "FDP3_left",
    "FDP4_left",
    "FDP5_left",
    "EDC2_left",
    "EDC3_left",
    "EDC4_left",
    "EDC5_left",
    "EDM_left",
    "EIP_left",
    "EPL_left",
    "EPB_left",
    "FPL_left",
    "APL_left",
    "OP_left",
    "RI2_left",
    "RI3_left",
    "RI4_left",
    "RI5_left",
    "LU_RB2_left",
    "LU_RB3_left",
    "LU_RB4_left",
    "LU_RB5_left",
    "UI_UB2_left",
    "UI_UB3_left",
    "UI_UB4_left",
    "UI_UB5_left",
)


FULLBODY_BODY2SITES_FOR_MIMIC = {
    "pelvis": "pelvis_mimic",
    "lumbar1": "upper_body_mimic",
    "head": "head_mimic",
    "humerus_l": "left_shoulder_mimic",
    "ulna_l": "left_elbow_mimic",
    "lunate_l": "left_hand_mimic",
    "humerus_r": "right_shoulder_mimic",
    "ulna_r": "right_elbow_mimic",
    "lunate_r": "right_hand_mimic",
    "femur_l": "left_hip_mimic",
    "tibia_l": "left_knee_mimic",
    "talus_l": "left_ankle_mimic",
    "toes_l": "left_toes_mimic",
    "femur_r": "right_hip_mimic",
    "tibia_r": "right_knee_mimic",
    "talus_r": "right_ankle_mimic",
    "toes_r": "right_toes_mimic",
}


def _prepare_scene_xml_keep_only_floor(scene_xml: Path) -> Path:
    """Materialize a scene XML with worldbody reduced to only floor geom.

    Args:
        scene_xml: Source scene XML path.

    Returns:
        Temporary XML path in the same directory as ``scene_xml``.
    """
    root = ET.fromstring(scene_xml.read_text(encoding="utf-8"))

    # Match MyoSuite floor appearance while keeping upstream floor physics.
    myosuite_floor_texture = Path()  # Keep the packaged floor texture; no MyoSuite runtime.
    use_myosuite_floor_texture = myosuite_floor_texture.is_file()

    # Remove MuscleMimic sky branding/colors and force a neutral skybox.
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for elem in list(asset):
        if elem.tag == "texture" and (elem.get("type") or "") == "skybox":
            asset.remove(elem)
            continue
        if (
            use_myosuite_floor_texture
            and elem.tag == "texture"
            and (elem.get("name") or "") == "texfloor"
        ):
            asset.remove(elem)
            continue
        if (
            use_myosuite_floor_texture
            and elem.tag == "material"
            and (elem.get("name") or "") == "matfloor"
        ):
            asset.remove(elem)
            continue
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sky",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0 0 0",
            "rgb2": "0 0 0",
            "width": "100",
            "height": "100",
        },
    )
    if use_myosuite_floor_texture:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "texfloor",
                "type": "2d",
                "height": "1",
                "width": "1",
                "file": myosuite_floor_texture.as_posix(),
            },
        )
        ET.SubElement(
            asset,
            "material",
            {
                "name": "matfloor",
                "reflectance": "0.01",
                "texture": "texfloor",
                "texrepeat": "1 1",
                "texuniform": "true",
            },
        )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0 0 0 0")

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.tag != "geom":
                worldbody.remove(child)
                continue
            geom_name = (child.get("name") or "").lower()
            geom_type = (child.get("type") or "").lower()
            is_floor = geom_name == "floor" or geom_type == "plane"
            if not is_floor:
                worldbody.remove(child)
                continue
            if use_myosuite_floor_texture:
                child.set("material", "matfloor")

    with tempfile.NamedTemporaryFile(
        "w",
        dir=scene_xml.parent,
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(ET.tostring(root, encoding="unicode"))
        return Path(tmp.name)


def _load_spec_with_floor_only_upstream_scene(xml_path: str) -> mujoco.MjSpec:
    """Load XML replacing upstream scene include with floor-only variant.

    Args:
        xml_path: Source model XML path.

    Returns:
        MuJoCo spec with the scene include rewritten to floor-only variant.
    """
    src = Path(xml_path)
    scene_temp_path: Path | None = None
    temp_model_path: Path | None = None
    try:
        model_root = ET.fromstring(src.read_text(encoding="utf-8"))
        include_rewritten = False
        for include_elem in model_root.findall("include"):
            include_file = include_elem.get("file")
            if include_file is None or "scene" not in include_file.lower():
                continue
            include_path = (src.parent / include_file).resolve()
            scene_temp_path = _prepare_scene_xml_keep_only_floor(include_path)
            include_elem.set("file", scene_temp_path.as_posix())
            include_rewritten = True
            break

        if not include_rewritten:
            return mujoco.MjSpec.from_file(src.as_posix())

        with tempfile.NamedTemporaryFile(
            "w",
            dir=src.parent,
            suffix=".xml",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            temp_model_path = Path(tmp.name)
            tmp.write(ET.tostring(model_root, encoding="unicode"))
        return mujoco.MjSpec.from_file(temp_model_path.as_posix())
    finally:
        if temp_model_path is not None and temp_model_path.exists():
            temp_model_path.unlink()
        if scene_temp_path is not None and scene_temp_path.exists():
            scene_temp_path.unlink()


def _apply_mimic_fullbody_spec_changes(
    spec: mujoco.MjSpec,
    disable_fingers: bool,
    finger_joint_tokens: tuple[str, ...],
    finger_muscle_tokens: tuple[str, ...],
) -> mujoco.MjSpec:
    """Apply the shared Mimic full-body edits (finger removal, sites, ctrlrange).

    Args:
        spec: Loaded, uncompiled full-body MjSpec.
        disable_fingers: Remove the finger joints, muscles and tendons
            (``config.disable_fingers``, see :func:`default_mimic_fullbody_config`).
        finger_joint_tokens: Finger joint names to remove when
            *disable_fingers*, in the naming convention of *spec*.
        finger_muscle_tokens: Finger muscle/tendon name substrings to remove,
            in the naming convention of *spec*.

    Returns:
        The same spec, edited in place.
    """
    if disable_fingers:
        joints_to_remove = [j for j in spec.joints if j.name in finger_joint_tokens]
        for joint in joints_to_remove:
            spec.delete(joint)

        actuators_to_remove = [
            a for a in spec.actuators if a.name in finger_muscle_tokens
        ]
        for actuator in actuators_to_remove:
            spec.delete(actuator)

        tendons_to_remove = [
            t for t in spec.tendons if any(m in t.name for m in finger_muscle_tokens)
        ]
        for tendon in tendons_to_remove:
            spec.delete(tendon)

    for body_name, site_name in FULLBODY_BODY2SITES_FOR_MIMIC.items():
        body = spec.body(body_name)
        body.add_site(
            name=site_name,
            group=4,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.075, 0.05, 0.025],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=[0.0, 0.0, 0.0],
        )

    for actuator in spec.actuators:
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
            actuator.ctrlrange = [-1.0, 1.0]
            actuator.ctrllimited = True

    return spec



def build_model(xml_path=None, model_version="1.0.5"):
    """Compile our bundled model with upstream finger/site/control edits."""
    if xml_path is None:
        xml_path = Path(__file__).resolve().parents[2] / "assets/robots/myofullbody/model/body/myofullbody.xml"
    spec = _load_spec_with_floor_only_upstream_scene(str(xml_path))
    spec = _apply_mimic_fullbody_spec_changes(spec, True, FINGER_JOINT_TOKENS, FINGER_MUSCLE_TOKENS)
    apply_models_version(spec, model_version)
    spec.memory = 16 * 2**20
    model = spec.compile()
    model.opt.timestep = 0.002
    if model.nu != 354:
        raise ValueError(f"Expected 354 muscle actuators, found {model.nu}")
    return model, spec
