"""USD rigid bodies for a handheld pump. Builder reused from our A4 b987934. Import after Isaac Sim starts."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import design as d

GENERATED = Path(__file__).resolve().parents[1] / "assets" / "generated"
WHITE, BLUE, LABEL = (0.92, 0.93, 0.95), (0.10, 0.30, 0.75), (0.95, 0.75, 0.15)
BLACK, RED, GREEN, GREY = (0.05, 0.05, 0.06), (0.85, 0.10, 0.08), (0.10, 0.65, 0.20), (0.45, 0.47, 0.50)


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

    def cylinder(self, z0, z1, radius, color, collide=True):
        self._n += 1
        shape = UsdGeom.Cylinder.Define(self.stage, f"/Part/cyl_{self._n}")
        shape.CreateAxisAttr("Z")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(z1 - z0))
        shape.AddTranslateOp().Set(Gf.Vec3d(0, 0, (z0 + z1) / 2))
        self._finish(shape.GetPrim(), color, collide)

    def box(self, center, size, yaw, color, collide=True, roll=0.0):
        self._n += 1
        shape = UsdGeom.Cube.Define(self.stage, f"/Part/box_{self._n}")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        shape.AddRotateZOp().Set(math.degrees(yaw))
        if roll:
            shape.AddRotateXOp().Set(math.degrees(roll))
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        self._finish(shape.GetPrim(), color, collide)

    def save(self) -> Path:
        self.stage.GetRootLayer().Save()
        return self.path

    def cylinder_x(self, x0, x1, radius, color, collide=True):
        self._n += 1
        shape = UsdGeom.Cylinder.Define(self.stage, f"/Part/cylx_{self._n}")
        shape.CreateAxisAttr("X")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(x1 - x0))
        shape.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, 0, 0))
        self._finish(shape.GetPrim(), color, collide)


def barrel():
    """Hollow barrel: 12 wall staves around a bore, a closed outlet end, end rings and two flat feet."""
    b = _Body("barrel", d.MASS["barrel"], d.FRICTION["barrel"], com=(0, 0, 0))
    L, R, w = d.BARREL_LENGTH, d.BARREL_RADIUS, d.BARREL_WALL
    n = 12
    width = 2 * R * math.tan(math.pi / n) + 0.0005
    for i in range(n):
        a = 2 * math.pi * i / n
        b.box((0, (R - w / 2) * math.cos(a), (R - w / 2) * math.sin(a)), (L, width, w), 0.0, WHITE,
              roll=a - math.pi / 2 or 1e-9)  # thin side radial
    b.cylinder_x(-L / 2, -L / 2 + 0.006, R, BLUE)  # closed outlet end
    b.cylinder_x(-L / 2 + 0.002, -L / 2 + 0.010, d.RING_RADIUS, BLUE, collide=False)
    b.cylinder_x(L / 2 - 0.010, L / 2 - 0.002, d.RING_RADIUS, BLUE, collide=False)
    b.cylinder_x(-L / 2 - 0.012, -L / 2, 0.005, BLACK, collide=False)  # outlet nipple (hose not simulated)
    fz = -d.AXIS_HEIGHT + d.FOOT[2] / 2
    for x in (-d.FOOT_X, d.FOOT_X):
        b.box((x, 0, fz), d.FOOT, 0.0, GREY)
    return b.save()


def piston():
    b = _Body("piston", d.MASS["piston"], d.FRICTION["piston"], com=(-.035, 0, 0))
    b.cylinder_x(-.190, -.009, .004, WHITE)
    b.box((0, 0, 0), (.018, .048, .032), 0, LABEL)
    return b.save()


def chamber():
    """Fixed gauge housing (test chamber) with a dial on top: 0-120 kPa over 270 degrees, black ticks every
    10 kPa (longer every 20), a solid green 54-66 kPa sector and a red 60 kPa target mark toward +x."""
    b = _Body("chamber", 1.0, (0.6, 0.5), kinematic=True)
    b.cylinder(0.0, d.CHAMBER_H, d.CHAMBER_R, GREY)
    b.cylinder(d.CHAMBER_H, d.CHAMBER_H + 0.002, d.DIAL_R, WHITE, collide=False)
    top = d.CHAMBER_H + 0.0025
    u = d.DIAL_R  # mark sizes scale with the dial
    lo, hi = d.TARGET_KPA - d.BAND_KPA, d.TARGET_KPA + d.BAND_KPA
    # acceptable band: a solid green sector at the rim (1 kPa slices)
    k = lo
    while k <= hi + 1e-9:
        a_ = d.needle_angle(k)
        r = 0.74 * u
        width = r * math.radians(d.GAUGE_SWEEP_DEG / d.GAUGE_MAX_KPA) * 1.6
        b.box((r * math.cos(a_), r * math.sin(a_), top), (0.36 * u, width, 0.001), a_, GREEN, collide=False)
        k += 1.0
    for k in range(0, int(d.GAUGE_MAX_KPA) + 1, d.TICK_MINOR):
        if lo <= k <= hi:
            continue
        a_ = d.needle_angle(k)
        major = k % d.TICK_MAJOR == 0
        ln = (0.3 if major else 0.18) * u
        r = 0.93 * u - ln / 2
        b.box((r * math.cos(a_), r * math.sin(a_), top), (ln, (0.075 if major else 0.05) * u, 0.001), a_, BLACK,
              collide=False)
    a_ = d.needle_angle(d.TARGET_KPA)  # target mark across the band
    b.box((0.74 * u * math.cos(a_), 0.74 * u * math.sin(a_), top + 0.0005), (0.4 * u, 0.06 * u, 0.001), a_, RED,
          collide=False)
    b.cylinder(d.CHAMBER_H + 0.002, d.CHAMBER_H + 0.006, 0.1 * d.DIAL_R, BLACK, collide=False)  # needle hub
    return b.save()


def needle():
    """Black gauge needle (display only, no collision): a kinematic pointer set from the gas model each step."""
    b = _Body("needle", 0.01, (0.5, 0.5), kinematic=True)
    u = d.DIAL_R
    b.box((0.4 * u, 0, 0), (0.8 * u, 0.07 * u, 0.0015), 0.0, BLACK, collide=False)
    b.box((-0.1 * u, 0, 0), (0.2 * u, 0.07 * u, 0.0015), 0.0, BLACK, collide=False)  # tail
    return b.save()


BUILDERS = {"barrel": barrel, "piston": piston, "chamber": chamber, "needle": needle}
