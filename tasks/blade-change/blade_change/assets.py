"""USD rigid bodies of the blade holder, collar, pin and blades (original simplified geometry). Import after Isaac Sim starts.

Body builder adapted from our A5 inspection-mount / A4 jar tasks."""

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
        body.CreateSolverPositionIterationCountAttr(32)
        body.CreateSolverVelocityIterationCountAttr(8)
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
            props.CreateContactOffsetAttr(0.002)
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

    def box(self, center, size, yaw, color, collide=True, tilt=0.0):
        self._n += 1
        shape = UsdGeom.Cube.Define(self.stage, f"/Part/box_{self._n}")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        shape.AddRotateZOp().Set(math.degrees(yaw))
        shape.AddRotateXOp().Set(math.degrees(tilt))
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        self._finish(shape.GetPrim(), color, collide)

    def save(self) -> Path:
        self.stage.GetRootLayer().Save()
        return self.path


BLUE, RED, WHITE = (0.12, 0.35, 0.80), (0.80, 0.15, 0.12), (0.95, 0.95, 0.95)


def holder() -> Path:
    """Channel for the tang (frame origin: nose-face centre on the axis), nose ring, rail shoe."""
    b = _Body("holder", d.MASS["holder"], d.FRICTION["holder"])
    L, hw, hh = d.HOLDER_LEN, d.HOLDER_HALF_W, d.HOLDER_HALF_H
    sx, sz = d.BLADE_T / 2 + d.SLOT_CLEAR, d.TANG_H / 2 + d.SLOT_CLEAR
    yc = L / 2
    m = d.MOUTH[2]
    b.box((0, (L + m) / 2, -(hh + sz) / 2), (2 * hw, L - m, hh - sz), 0.0, GREY)  # floor, behind the mouth
    b.box((0, (L + m) / 2, (hh + sz) / 2), (2 * hw, L - m, hh - sz), 0.0, GREY)  # roof (pin bore not modelled: pin-holder contact off)
    for s in (-1, 1):
        b.box((s * (hw + sx) / 2, (L + m) / 2, 0), (hw - sx, L - m, 2 * sz), 0.0, GREY)  # side walls behind the mouth
    # entry chamfer: four tapered plates from the mouth (MOUTH at the nose face) down to the slot at depth m
    mx, mz = d.MOUTH[0], d.MOUTH[1]
    ax, az = math.atan2(mx - sx, m), math.atan2(mz - sz, m)
    for s in (-1, 1):
        _plate(b, (s * ((mx + sx) / 2 + 0.002), m / 2, 0), (0.004, m / math.cos(ax), 2 * mz), yaw=s * ax)  # wide at the nose
        _plate(b, (0, m / 2, s * ((mz + sz) / 2 + 0.002)), (2 * mx + 0.008, m / math.cos(az), 0.004), tilt=-s * az)
    b.box((0, L - 0.002, 0), (2 * sx, 0.004, 2 * sz), 0.0, DARK)  # back of the channel
    b.box((0, 0.020, -0.045), (0.05, 0.08, 0.008), 0.0, DARK, collide=False)  # carriage shoe on its rail (visual)
    b.box((0, 0.020, -0.028), (0.012, 0.012, 0.026), 0.0, DARK, collide=False)  # post (visual)
    return b.save()


def _plate(b, center, size, yaw=0.0, tilt=0.0):
    b.box(center, size, yaw, GREY, tilt=tilt)


def collar() -> Path:
    """Ring at the nose (visual), lever tab on top, cam peg on the +x side reaching back under the pin's follower."""
    b = _Body("collar", d.MASS["collar"], d.FRICTION["collar"])
    _ring(b)
    r0 = d.COLLAR_R + 0.002
    b.box((0, 0, (r0 + d.LEVER_R) / 2), (d.LEVER[0], d.LEVER[1], d.LEVER_R - r0), 0.0, ORANGE)  # lever tab
    phi = math.radians(d.CAM_PHI_DEG)
    px, pz = d.CAM_R * math.sin(phi), d.CAM_R * math.cos(phi)
    y1 = d.PIN_Y - d.COLLAR_Y + 0.006
    b.box((px, y1 / 2, pz), (2 * d.PEG_R, y1, 2 * d.PEG_R), 0.0, ORANGE)  # cam peg (square bar)
    return b.save()


def _ring(b):
    for i in range(12):
        _ring_seg(b, 2 * math.pi * i / 12)


def _ring_seg(b, a):
    b._n += 1
    shape = UsdGeom.Cube.Define(b.stage, f"/Part/ring_{b._n}")
    shape.CreateSizeAttr(1.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(d.COLLAR_R * math.sin(a), 0.0, d.COLLAR_R * math.cos(a)))
    shape.AddRotateYOp().Set(math.degrees(a))
    shape.AddScaleOp().Set(Gf.Vec3f(0.012, 0.012, 0.005))
    b._finish(shape.GetPrim(), ORANGE, collide=False)


def _cyl_y(b, y0, y1, r, xz, color):
    b._n += 1
    shape = UsdGeom.Cylinder.Define(b.stage, f"/Part/cyl_{b._n}")
    shape.CreateAxisAttr("Y")
    shape.CreateRadiusAttr(float(r))
    shape.CreateHeightAttr(float(y1 - y0))
    shape.AddTranslateOp().Set(Gf.Vec3d(float(xz[0]), (y0 + y1) / 2, float(xz[1])))
    b._finish(shape.GetPrim(), color)


def pin() -> Path:
    """Square locking pin (bottom at PIN_ENGAGE above the axis when engaged) with its follower arm on top."""
    b = _Body("pin", d.MASS["pin"], d.FRICTION["pin"])
    z0 = d.PIN_ENGAGE
    zf = d.follower_bottom_rest()
    b.box((0, 0, (z0 + zf + 0.010) / 2), (d.PIN_W, d.PIN_W, zf + 0.010 - z0), 0.0, RED)
    b.box((d.FOLLOWER_X / 2, 0, zf + 0.005), (d.FOLLOWER_X + 0.006, 0.012, 0.010), 0.0, RED)
    return b.save()


def _blade(name, body_color, mark_color):
    """Frame origin: shoulder centre on the axis. Tang toward +y with the top-edge notch; body toward -y."""
    b = _Body(name, d.MASS[name], d.FRICTION[name])
    t, th, bh = d.BLADE_T, d.TANG_H, d.BODY_H
    low_h = th - d.NOTCH_D
    b.box((0, d.TANG_L / 2, -th / 2 + low_h / 2), (t, d.TANG_L, low_h), 0.0, DARK)  # tang below the notch
    n0, n1 = d.NOTCH_Y - d.NOTCH_W / 2, d.NOTCH_Y + d.NOTCH_W / 2
    for y0, y1 in ((0.0, n0), (n1, d.TANG_L)):
        b.box((0, (y0 + y1) / 2, th / 2 - d.NOTCH_D / 2), (t, y1 - y0, d.NOTCH_D), 0.0, DARK)
    b.box((0, -d.BODY_L / 2, 0), (t, d.BODY_L, bh), 0.0, body_color)  # body (shoulder at y = 0)
    b.box((0, -d.GRIP_FROM_SHOULDER, 0), d.GRIP_BLOCK, 0.0, body_color)  # grip block
    b.box((0, -d.BODY_L / 2, -bh / 2 + 0.002), (t + 0.0006, d.BODY_L - 0.01, 0.004), 0.0, BLACK, collide=False)  # blunt teeth edge
    b.box((0, -d.BODY_L + 0.03, bh / 4), (t + 0.0008, 0.03, 0.006), 0.0, mark_color, collide=False)  # identity mark
    return b.save()


def old_blade() -> Path:
    return _blade("old_blade", GREY, RED)


def new_blade() -> Path:
    return _blade("new_blade", BLUE, WHITE)


BUILDERS = {"holder": holder, "collar": collar, "pin": pin, "old_blade": old_blade, "new_blade": new_blade}
