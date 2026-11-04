"""USD rigid bodies of the inspection mount (original simplified geometry). Import after Isaac Sim starts."""

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


def post() -> Path:
    """Bench post (kinematic, fixed): a column, the hinge axle stub and the knob seat."""
    b = _Body("post", 1.0, d.FRICTION["post"], kinematic=True)
    h = d.POST_TOP - d.TOP
    b.box((0, 0, h / 2), (d.POST_SIZE[0], d.POST_SIZE[1], h), 0.0, GREY)
    hz = d.HINGE[2] - d.TOP
    b.cylinder(-0.045 + 0.016, -0.015, 0.008, DARK, collide=False, axis="X", at=(0.0, hz))  # axle (visual)
    return b.save()


def arm() -> Path:
    """Arm frame origin on the hinge axis; the bar runs along -y to the head."""
    m = d.MASS["arm"] + d.MASS["head"]
    com = (0.0, -(d.MASS["arm"] * d.ARM_L / 2 + d.MASS["head"] * d.ARM_L) / m, 0.0)
    b = _Body("arm", m, d.FRICTION["arm"], com=com)
    b.cylinder(-d.BAR_W / 2 - 0.004, d.BAR_W / 2 + 0.004, d.HUB_R, DARK, axis="X", at=(0.0, 0.0))
    L = d.ARM_L - d.HEAD[1] / 2
    b.box((0, -L / 2, 0), (d.BAR_W, L, d.BAR_T), 0.0, GREY)
    b.box((0, -d.ARM_L, 0), tuple(d.HEAD), 0.0, BLACK)  # inspection head
    b.box((0, -d.ARM_L - d.HEAD[1] / 2 - 0.006, 0), (0.024, 0.012, 0.024), 0.0, ORANGE, collide=False)  # lens
    return b.save()


def knob() -> Path:
    b = _Body("knob", d.MASS["knob"], d.FRICTION["knob"], com=(0, 0, 0))
    # ridged round knob: 16 thin radial slabs through the axis, 11.25 degrees apart (box colliders);
    # outer radius 20.0-20.04 mm, with shallow ridges between the slab ends
    n = 16
    w = 2 * d.KNOB_R * math.tan(math.pi / (2 * n))
    for i in range(n):
        b.box((0, 0, 0), (2 * d.KNOB_R, w, d.KNOB_H), math.pi / n * i, ORANGE)
    b.cylinder(-d.KNOB_H / 2 - d.KNOB_LIFT, -d.KNOB_H / 2, 0.006, DARK, collide=False)  # threaded stem (visual)
    b.box((0, d.KNOB_R * 0.55, d.KNOB_H / 2 + 0.0005), (0.004, 0.012, 0.001), 0.0, BLACK, collide=False)  # index mark
    return b.save()


BUILDERS = {"post": post, "arm": arm, "knob": knob}
