"""USD rigid bodies of the puller fixture (original simplified geometry). Body builder copied from our A5 task
(f7e6030), itself from A4. Import after Isaac Sim starts."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import design as d

GENERATED = Path(__file__).resolve().parents[1] / "assets" / "generated"
GREY, DARK, ORANGE, BLACK = (0.62, 0.64, 0.67), (0.25, 0.27, 0.30), (0.95, 0.50, 0.10), (0.08, 0.08, 0.09)


def _linear(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


class _Body:
    def __init__(self, name, mass, friction, com=None, kinematic=False):
        GENERATED.mkdir(parents=True, exist_ok=True)
        self.path = GENERATED / f"{name}.usda"
        self.stage = Usd.Stage.CreateNew(str(self.path)) if not self.path.exists() else Usd.Stage.Open(str(self.path))
        self.stage.GetRootLayer().Clear()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        xf = UsdGeom.Xform.Define(self.stage, "/Part")
        self.stage.SetDefaultPrim(xf.GetPrim())
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(bool(kinematic))
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(mass))
        if com is not None:
            m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSolverPositionIterationCountAttr(16)
        body.CreateSolverVelocityIterationCountAttr(4)
        body.CreateMaxDepenetrationVelocityAttr(0.3)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.physics = UsdShade.Material.Define(self.stage, "/Part/physics_material")
        pm = UsdPhysics.MaterialAPI.Apply(self.physics.GetPrim())
        pm.CreateStaticFrictionAttr(friction[0])
        pm.CreateDynamicFrictionAttr(friction[1])
        pm.CreateRestitutionAttr(0.0)
        self._looks, self._n = {}, 0

    def _look(self, color):
        if color not in self._looks:
            mat = UsdShade.Material.Define(self.stage, f"/Part/look_{len(self._looks)}")
            sh = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("shader"))
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_linear(color)))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self._looks[color] = mat
        return self._looks[color]

    def _finish(self, prim, color, collide=True):
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            props = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            props.CreateContactOffsetAttr(0.001)
            props.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.physics, materialPurpose="physics")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._look(color))

    def cylinder(self, z0, z1, radius, color, collide=True, axis="Z", at=(0.0, 0.0)):
        self._n += 1
        shape = UsdGeom.Cylinder.Define(self.stage, f"/Part/cyl_{self._n}")
        shape.CreateAxisAttr(axis)
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(z1 - z0))
        c = [*at, (z0 + z1) / 2] if axis == "Z" else [(z0 + z1) / 2, *at]  # X axis: at = (y, z)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        self._finish(shape.GetPrim(), color, collide)

    def box(self, center, size, yaw, color, collide=True):
        self._n += 1
        shape = UsdGeom.Cube.Define(self.stage, f"/Part/box_{self._n}")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        shape.AddRotateZOp().Set(math.degrees(yaw))
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        self._finish(shape.GetPrim(), color, collide)

    def save(self) -> Path:
        self.stage.GetRootLayer().Save()
        return self.path


def _ring(b, z0, z1, r_in, r_out, color, n=16):
    w = 2 * r_out * math.tan(math.pi / n) + 0.0005
    for k in range(n):
        a = 2 * math.pi * k / n
        r = (r_in + r_out) / 2
        b.box((r * math.cos(a), r * math.sin(a), (z0 + z1) / 2), (r_out - r_in, w, z1 - z0), a, color)


def fixture() -> Path:
    """Fixed base flange and vertical shaft (kinematic)."""
    b = _Body("fixture", 1.0, (0.6, 0.5), kinematic=True)
    b.box((0, 0, d.FLANGE[2] / 2), d.FLANGE, 0.0, GREY)
    b.cylinder(d.FLANGE[2], d.SHAFT_TOP, d.SHAFT_R, DARK)
    _ring(b, d.SHAFT_TOP, d.SHAFT_TOP + d.CUP_H, d.CUP_R, d.SHAFT_R, DARK, n=12)  # centre hole rim of the shaft end
    return b.save()


def collar() -> Path:
    b = _Body("collar", d.MASS["collar"], (0.6, 0.5), com=(0, 0, 0.008))
    _ring(b, 0.0, d.HUB_H, d.COLLAR_BORE, d.HUB_R, ORANGE)
    _ring(b, d.HUB_H, d.HUB_H + d.RIM_H, d.COLLAR_BORE, d.COLLAR_R, ORANGE)
    return b.save()


def body() -> Path:
    b = _Body("body", d.MASS["body"], (0.8, 0.7), com=(0, 0, 0.015))
    b.box((0, 0, 0), d.CROSSBAR, 0.0, BLACK)
    b.box((0, 0, d.CROSSBAR[2] / 2 + d.BOSS[2] / 2), d.BOSS, 0.0, GREY)
    # side handle: the holding hand stays 11 cm from the axis, clear of the jaws and the knob (arms collided, draft-008)
    b.box((0, -d.BOSS[1] / 2 - d.HANDLE[1] / 2, d.HANDLE_Z), d.HANDLE, 0.0, BLACK)
    # grip block at the end: pinched across y, so neither the tool's weight (about x) nor the turning reaction
    # (about z) acts on the pinch's free axis (a vertical pinch let the tool yaw 1 cm off the axis, draft-010)
    b.box((0, -d.BOSS[1] / 2 - d.HANDLE[1] - d.GRIP_BLOCK[1] / 2 + 0.005, d.GRIP_Z), d.GRIP_BLOCK, 0.0, GREY)
    return b.save()


def handle() -> Path:
    b = _Body("handle", d.MASS["handle"], (1.0, 0.9), com=(0, 0, 0))
    for k in range(8):  # ridged knob from radial slabs (A1 lesson: solid, no star corners)
        b.box((0, 0, 0), (2 * d.KNOB_R, 2 * d.KNOB_R * math.tan(math.pi / 16), 0.020), math.pi / 8 * k, ORANGE)
    b.box((0, 0.010, 0.0105), (0.004, 0.010, 0.001), 0.0, BLACK, collide=False)  # index mark
    b.cylinder(-0.014, -0.010, 0.005, DARK, collide=False)  # neck (visual)
    return b.save()


def ram() -> Path:
    b = _Body("ram", d.MASS["ram"], (0.6, 0.5), com=(0, 0, 0))
    b.cylinder(d.RAM_TIP + 0.003, d.CROSSBAR[2] / 2 - 0.001, 0.004, DARK, collide=False)  # inside the body: visual
    b.box((0, 0, d.RAM_TIP + 0.0015), (0.008, 0.008, 0.003), 0.0, DARK)  # tip pad
    b.cylinder(d.RAM_TIP - d.POINT_L, d.RAM_TIP, d.POINT_R, DARK)  # point: seats in the shaft's centre hole
    b.cylinder(d.RAM_TIP + 0.003, -d.CROSSBAR[2] / 2 - 0.0005, 0.004, DARK)  # exposed spindle (collides)
    return b.save()


def jaw(sign) -> Path:
    b = _Body("jaw_pos" if sign > 0 else "jaw_neg", d.MASS["jaw"], (0.9, 0.8), com=(0, 0, -0.03))
    s = float(sign)
    # (no tab above the crossbar: the holder's fingers sweep that space when pinching the boss; draft-001)
    top, bot = -d.CROSSBAR[2] / 2 - 0.002, d.HOOK_Z + 0.003
    b.box((0, 0, (top + bot) / 2), (0.006, 0.014, top - bot), 0.0, GREY)  # arm
    b.box((-s * d.HOOK_IN / 2, 0, d.HOOK_Z + 0.0015), (d.HOOK_IN + 0.006, 0.014, 0.003), 0.0, GREY)  # hook lip
    return b.save()


BUILDERS = {"fixture": fixture, "collar": collar, "body": body, "handle": handle, "ram": ram,
            "jaw_pos": lambda: jaw(1), "jaw_neg": lambda: jaw(-1)}
