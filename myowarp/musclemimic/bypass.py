"""Whole-body CMU bent-knee bypass assembly ported from prosim/bypass.py."""
from __future__ import annotations
import os
from pathlib import Path
import mujoco
import numpy as np
from .model import build_model

ROOT = Path(__file__).resolve().parents[2]
MODELS = str(ROOT / "assets/robots/myofullbody_cmu_leg")
KNEE_LOCK_DEG = 90.0
FOOT_LOCK_DEG = {"ankle_angle_r": 0.0, "subtalar_angle_r": 0.0, "mtp_angle_r": 0.0}
LOCK_SOLREF = (0.005, 1.0)
NO_CONTACT_BODIES = ("calcn_r", "toes_r")
LEFT_FOOT_BODIES = ("calcn_l", "toes_l")
FRAME_MASS = 0.0
PROSTHESIS_CONTACTS = ("toe", "heel")
PROSTHESIS_JOINTS = ("knee_motor", "knee", "ankle_motor", "ankle_drive")

def fullbody_spec():
    return build_model()[1]

BASES = {"fullbody": fullbody_spec}

def plain_scene(spec: mujoco.MjSpec) -> None:
    """Swap MyoSuite's room (wall/prop mesh, logo, wood floor) for the stock
    MuJoCo look: gradient skybox and a checker floor.  Visual only -- the
    floor geom and its contact properties are unchanged.  Works on any spec
    whose ground geom is named "floor" (myoLeg, MyoFullBody)."""
    for g in list(spec.worldbody.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:      # the room and the logo
            spec.delete(g)
    floor_mat = spec.geom("floor").material or "plain_floor"
    if not spec.material(floor_mat):
        spec.add_material(name=floor_mat)
        spec.geom("floor").material = floor_mat
    spec.add_texture(name="plain_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="plain_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.2, 0.3, 0.4], width=300, height=300)
    mat = spec.material(floor_mat)
    mat.textures = [""] * len(mat.textures)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "plain_grid"
    mat.texrepeat, mat.texuniform, mat.reflectance = [5, 5], True, 0.2
    mat.rgba = [1, 1, 1, 1]


def lock_right_leg(spec: mujoco.MjSpec, knee_deg: float = KNEE_LOCK_DEG,
                   foot_deg: dict[str, float] | None = None) -> None:
    """Hold the right knee and foot joints at fixed angles.

    A joint equality with no second joint pins joint1 to polycoef[0].  The
    knee's coupled joints are left alone: their existing equalities follow
    `knee_angle_r` to wherever it is held.
    """
    locks = {"knee_angle_r": knee_deg, **(foot_deg or FOOT_LOCK_DEG)}
    for joint, deg in locks.items():
        eq = spec.add_equality()
        eq.name = f"bypass_lock_{joint}"
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = joint
        eq.data[:5] = [np.radians(deg), 0, 0, 0, 0]
        eq.solref = LOCK_SOLREF
    for body in NO_CONTACT_BODIES:
        for g in spec.body(body).geoms:
            g.contype = g.conaffinity = 0


def attach_prosthesis(spec: mujoco.MjSpec, hip_to_knee: float,
                      xml: str = os.path.join(MODELS, "prosthesis_source.xml")) -> None:
    """Weld the leg to femur_r with its knee axis `hip_to_knee` below the hip
    centre, down the femur's long axis.  The transform is fixed at attach
    time (editing the frame afterwards does nothing), so the height has to be
    known first.  The child's <option> (0.2 ms, y-up gravity) is discarded
    by attach; the parent's is kept."""
    child = mujoco.MjSpec.from_file(xml)
    # Match the child's <option> to the parent's so attach has nothing to
    # warn about; the parent's values win either way.
    for opt in ("timestep", "gravity", "integrator"):
        setattr(child.option, opt, getattr(spec.option, opt))
    meshdir = str(ROOT / "assets/robots/cmu_leg/meshes")
    for mesh in child.meshes:       # resolve against the leg's meshdir, not myoLeg's
        mesh.file = os.path.join(meshdir, os.path.basename(mesh.file))
    # myoLeg convention: body geoms collide with the floor only (contype 1,
    # conaffinity 0), never with each other.
    for name in PROSTHESIS_CONTACTS:
        child.geom(name).conaffinity = 0
    frame = spec.body("femur_r").add_frame()
    frame.pos = [0.0, -hip_to_knee, 0.0]
    frame.attach_body(child.body("kneeStator"), "", "")


def _floor_gap(m, d, bodies=(), geoms=()) -> float:
    """Smallest signed distance from the floor to any colliding geom on
    `bodies` or named in `geoms` (negative = penetrating)."""
    floor = m.geom("floor").id
    ids = [g for g in range(m.ngeom) if m.geom_contype[g] and g != floor
           and (m.body(m.geom_bodyid[g]).name in bodies or m.geom(g).name in geoms)]
    return min(mujoco.mj_geomDistance(m, d, floor, g, 1.0, None) for g in ids)


def _neutral(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """qpos0 with the locks and knee couplings applied."""
    d.qpos[:] = m.qpos0
    _apply_couplings(m, d)          # locks first, then the knee couplings
    _apply_couplings(m, d)
    mujoco.mj_forward(m, d)


def _apply_couplings(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """Put every two-joint equality's follower on its polynomial, so a pose
    written into qpos starts on the constraint manifold instead of snapping."""
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        a1 = m.jnt_qposadr[m.eq_obj1id[i]]
        if m.eq_obj2id[i] < 0:
            d.qpos[a1] = m.eq_data[i][0]
        else:
            x = d.qpos[m.jnt_qposadr[m.eq_obj2id[i]]]
            d.qpos[a1] = sum(m.eq_data[i][k] * x**k for k in range(5))


def _leg_mismatch(spec: mujoco.MjSpec) -> float:
    """How much lower the prosthetic sole sits than the left foot in the
    neutral pose, with the hip fixed (m; positive = prosthesis longer)."""
    m = spec.compile()
    d = mujoco.MjData(m)
    _neutral(m, d)
    return (_floor_gap(m, d, bodies=LEFT_FOOT_BODIES)
            - _floor_gap(m, d, geoms=PROSTHESIS_CONTACTS))


def stand_pose(m: mujoco.MjModel) -> np.ndarray:
    """qpos for standing in the bypass posture: qpos0 (the prosthesis at its
    as-built pose, where the four-bar closes) plus the locks, root lowered
    until the lowest foot touches the floor."""
    d = mujoco.MjData(m)
    _neutral(m, d)
    d.qpos[2] -= _floor_gap(m, d, LEFT_FOOT_BODIES, PROSTHESIS_CONTACTS)
    return d.qpos.copy()


def _assemble(base, knee_deg, foot_deg, hip_to_knee, scene, implicit,
              lock_prosthesis) -> mujoco.MjSpec:
    spec = BASES[base]()
    if scene == "plain":
        plain_scene(spec)
    if implicit:
        # The leg's velocity loop is stiff; integrate it implicitly, as the
        # standalone model does.  (Off keeps the base model's own integrator,
        # e.g. to run a policy on the physics it was trained on.)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Keep one key and overwrite it: in MuJoCo 3.11, deleting the LAST key
    # leaves a phantom keyframe in the compiled model.
    while len(spec.keys) > 1:
        spec.delete(spec.keys[-1])
    if not spec.keys:
        spec.add_key()
    lock_right_leg(spec, knee_deg, foot_deg)
    if hip_to_knee is not None:
        attach_prosthesis(spec, hip_to_knee)
        if lock_prosthesis:
            for joint in PROSTHESIS_JOINTS:
                eq = spec.add_equality()
                eq.name, eq.type, eq.name1 = f"lock_{joint}", mujoco.mjtEq.mjEQ_JOINT, joint
                eq.data[:5] = [0, 0, 0, 0, 0]
                eq.solref = LOCK_SOLREF
        if FRAME_MASS > 0:
            spec.body("femur_r").add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0],
                pos=[0, -(hip_to_knee - 0.0775), 0], mass=FRAME_MASS,
                contype=0, conaffinity=0, group=3)
    return spec


def build(knee_deg: float = KNEE_LOCK_DEG,
          foot_deg: dict[str, float] | None = None,
          hip_to_knee: float | None = None, prosthesis: bool = True,
          scene: str = "plain", base: str = "fullbody", implicit: bool = True,
          lock_prosthesis: bool = False):
    """Return (spec, model): myoLeg locked in the bypass posture, the
    prosthesis welded on, one keyframe `bypass_stand`.  myoLeg's own
    keyframes are dropped: they put the right knee where the lock no longer
    allows.

    hip_to_knee=None levels the legs (see module docstring); pass a value to
    override.  The value used is recorded in the custom numeric
    `bypass_hip_to_knee`, and the remaining leg-length mismatch (prosthetic
    minus biological, m) in `bypass_leg_mismatch`.  scene="plain" (default)
    uses the stock MuJoCo look; "myosuite" keeps MyoSuite's room.

    base: "myoleg" (80 muscles) or "fullbody" (MuscleMimic's MyoFullBody,
    354).  implicit=False keeps the base's own integrator.  lock_prosthesis
    holds the leg's knee and ankle rigid at the as-built pose (a stilt)."""
    args = (base, knee_deg, foot_deg)
    rest = (scene, implicit, lock_prosthesis)
    if not prosthesis:
        spec = _assemble(*args, None, *rest)
    else:
        h = hip_to_knee
        if h is None:
            # Slide the mount along the femur's long axis until the soles are
            # level.  With the femur vertical in the neutral pose (myoLeg) one
            # trial build is exact; iterate in case a base's femur is tilted.
            h = 0.5
            for _ in range(4):
                miss = _leg_mismatch(_assemble(*args, h, *rest))
                if abs(miss) < 1e-6:
                    break
                h -= miss
        spec = _assemble(*args, h, *rest)
        spec.add_numeric(name="bypass_hip_to_knee", data=[h])
        spec.add_numeric(name="bypass_leg_mismatch", data=[_leg_mismatch(spec)])
    m = spec.compile()
    key = spec.keys[0]
    key.name = "bypass_stand"
    key.qpos = stand_pose(m)
    return spec, spec.compile()

