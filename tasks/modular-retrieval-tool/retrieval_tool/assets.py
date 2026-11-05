"""USD rigid bodies of the modular retrieval tool task (original simplified geometry; box/convex colliders).
Builder copied from our A6 hand pump (3a11129). Import after Isaac Sim starts."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import design as d

GENERATED = Path(__file__).resolve().parents[1] / "assets" / "generated"
STEEL, DARK, ORANGE, BLUE = (0.72, 0.74, 0.77), (0.20, 0.21, 0.23), (0.95, 0.50, 0.10), (0.12, 0.35, 0.80)
GREY, GREEN, WHITE = (0.55, 0.57, 0.60), (0.20, 0.60, 0.25), (0.92, 0.93, 0.95)


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
        # The release linkage needs accurate guiding under grasp and gate contact.
        # Resolve its guide accurately without changing applied forces.
        guide_body = name in ("head", "plunger", "shaft")
        body.CreateSolverPositionIterationCountAttr(128 if guide_body else 16)
        body.CreateSolverVelocityIterationCountAttr(8 if guide_body else 4)
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

    def material(self, name, friction):
        """An extra physics material for some shapes of this body (set .use before adding them)."""
        mat = UsdShade.Material.Define(self.stage, f"/Part/{name}")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(friction[0])
        pm.CreateDynamicFrictionAttr(friction[1])
        pm.CreateRestitutionAttr(0.0)
        return mat

    def _finish(self, prim, color, collide=True):
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            props = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            props.CreateContactOffsetAttr(0.001)
            props.CreateRestOffsetAttr(0.0)
            mat = getattr(self, "use", None) or self.physics
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
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


def _box(b, x0, x1, y0, y1, z0, z1, color, collide=True, yaw=0.0):
    """Axis-aligned box from bounds (in the part frame), optionally turned about z through the origin."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c, s = math.cos(yaw), math.sin(yaw)
    b.box((c * cx - s * cy, s * cx + c * cy, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), yaw, color, collide)


def shaft():
    b = _Body("shaft", d.MASS["shaft"], d.FRICTION["shaft"], com=(d.SHAFT_COM_X, 0, 0.012))
    _box(b, -d.ROD_L, 0.008, -d.ROD_W / 2, d.ROD_W / 2, 0, d.ROD_T, STEEL)  # rod (with a boss around the pin)
    b.use = b.material("grip_rubber", d.HANDLE_FRICTION)
    _box(b, -d.ROD_L - d.HANDLE_L, -d.ROD_L, -d.HANDLE_W / 2, d.HANDLE_W / 2, 0, d.HANDLE_H, DARK)  # handle
    b.use = None
    top = d.ROD_T + d.PLATE_T + d.LUG_GAP + d.LUG_T
    b.cylinder(d.ROD_T, top, d.PIN_R, STEEL)  # bayonet pin
    z0 = d.ROD_T + d.PLATE_T + d.LUG_GAP
    _box(b, -d.LUG_W / 2, d.LUG_W / 2, -d.LUG_R, d.LUG_R, z0, top, STEEL)  # two lugs (along y)
    return b.save()


def head():
    b = _Body("head", d.MASS["head"], d.FRICTION["head"])
    P, H, T = d.PLATE_HALF, d.HOLE_HALF, d.PLATE_T
    # keyed plate: square hole, slots along x for the lugs
    _box(b, -P, P, H, P, 0, T, ORANGE)
    _box(b, -P, P, -P, -H, 0, T, ORANGE)
    for sx in (1, -1):
        lo, hi = sorted((sx * H, sx * P))
        _box(b, lo, hi, d.SLOT_HALF_W, H, 0, T, ORANGE)
        _box(b, lo, hi, -H, -d.SLOT_HALF_W, 0, T, ORANGE)
        lo2, hi2 = sorted((sx * d.SLOT_END, sx * P))
        _box(b, lo2, hi2, -d.SLOT_HALF_W, d.SLOT_HALF_W, 0, T, ORANGE)
    # bell: walls and roof around the lug cavity
    z0, z1 = T, T + d.BELL_CAVITY
    w = d.BELL_WALL
    # Gate window lies opposite the hook's insertion-to-lock sweep.
    _box(b, -P, 0.007, P-w, P, z0, z1, ORANGE)
    _box(b, 0.017, P, P-w, P, z0, z1, ORANGE)
    _box(b, 0.007, 0.017, P-w, P, z0, 0.007, ORANGE)
    _box(b, 0.007, 0.017, P-w, P, 0.015, z1, ORANGE)
    _box(b, -P, P, -P, -P+w, z0, z1, ORANGE)
    _box(b, P - w, P, -P + w, P - w, z0, z1, ORANGE)
    _box(b, -P, -P + w, -P + w, P - w, z0, z1, ORANGE)
    _box(b, -P, P, -P, P, z1, z1 + d.BELL_ROOF, ORANGE)
    nx, ny, nz = d.NECK
    _box(b, -nx / 2, nx / 2, -ny / 2, ny / 2, z1 + d.BELL_ROOF, z1 + d.BELL_ROOF + nz, DARK)  # grip neck
    # stop blocks inside the cavity end the quarter turn
    for a in (d.STOP_ANGLE, d.STOP_ANGLE + math.pi):
        r = d.STOP_R
        b.box((r * math.cos(a), r * math.sin(a), (z0 + z1) / 2), (0.004, 0.004, z1 - z0), a, ORANGE)
    # hook arm and downward tooth
    _box(b, d.ARM_X0, d.ARM_X1, -d.ARM_W / 2, d.ARM_W / 2, 0, d.ARM_H, ORANGE)
    _box(b, d.TOOTH_X0, d.ARM_X1, -d.ARM_W / 2, d.ARM_W / 2, -d.TOOTH_DROP, 0, ORANGE)
    return b.save()


def cartridge():
    b = _Body("cartridge", d.MASS["cartridge"], d.FRICTION["cartridge"])
    L, W, Hc, t = d.CART_L, d.CART_W / 2, d.CART_H, d.CART_BASE_T
    _box(b, 0, L, -W, W, 0, t, BLUE)  # base
    _box(b, 0, d.POCKET_X0, -W, W, t, Hc, BLUE)  # front lip
    _box(b, d.POCKET_X1, L, -W, W, t, d.CART_BODY_H, BLUE)  # body behind the pocket (grip)
    for s in (1, -1):
        lo, hi = sorted((s * d.POCKET_HALF_W, s * W))
        _box(b, d.POCKET_X0, d.POCKET_X1, lo, hi, t, Hc, BLUE)
    _box(b, 0.035, 0.055, -0.012, 0.012, d.CART_BODY_H, d.CART_BODY_H + 0.0005, WHITE, collide=False)  # label
    return b.save()


def hood():
    """Fixed service hood; frame origin at the mouth centre on the table. Interior x 0..depth, |y| < half w."""
    b = _Body("hood", 2.0, (0.5, 0.45), kinematic=True)
    D, Wh, Hh, t = d.HOOD_DEPTH, d.HOOD_HALF_W, d.HOOD_H, d.HOOD_WALL
    _box(b, 0, D + t, Wh, Wh + t, 0, Hh, GREY)
    _box(b, 0, D + t, -Wh - t, -Wh, 0, Hh, GREY)
    _box(b, D, D + t, -Wh, Wh, 0, Hh, GREY)  # closed rear
    _box(b, 0, D + t, -Wh - t, Wh + t, Hh, Hh + t, GREY)  # roof
    _box(b, -0.002, 0, -Wh - t, Wh + t, Hh, Hh + t, DARK, collide=False)  # mouth edge (visual)
    return b.save()


def tray():
    b = _Body("tray", 1.0, (0.5, 0.45), kinematic=True)
    ix, iy = d.TRAY_INNER[0] / 2, d.TRAY_INNER[1] / 2
    t, h = d.TRAY_RIM
    _box(b, -ix - t, ix + t, -iy - t, iy + t, 0, 0.003, GREEN)
    _box(b, -ix - t, ix + t, iy, iy + t, 0, h, GREEN)
    _box(b, -ix - t, ix + t, -iy - t, -iy, 0, h, GREEN)
    _box(b, ix, ix + t, -iy, iy, 0, h, GREEN)
    _box(b, -ix - t, -ix, -iy, iy, 0, h, GREEN)
    return b.save()


def stand():
    b = _Body("stand", 1.0, (0.5, 0.45), kinematic=True)
    sx, sy, sz = d.HEAD_STAND
    _box(b, -sx / 2, sx / 2, -sy / 2, sy / 2, 0, sz, GREY)
    return b.save()


def plunger():
    """One rigid squeeze control/linkage/gate, guided laterally on the loose shaft."""
    b = _Body("plunger", d.RELEASE_MASS, (0.5, 0.4))
    def link(x0, x1, y0, y1, z0, z1, color):
        # Mirror away from the hook sweep and tuck the forward linkage inside
        # the existing 50 mm hood opening (outer edge +23 mm).
        _box(b, x0, x1, -y1-0.006, -y0-0.006, z0, z1, color)
    # Trigger face beside handle. Rear bridge and side link clear the fixed handle.
    link( -0.190, -0.158, 0.033, 0.039, 0.010, 0.034, BLUE)
    # Rear handle remains available for a tool-use regrasp with release shut.
    link( -0.252, -0.190, 0.033, 0.039, 0.0005, 0.0035, BLUE)
    link( -0.194, -0.190, 0.033, 0.039, 0.0035, 0.034, BLUE)
    link( -0.256, -0.250, -0.029, 0.039, 0.0005, 0.0035, STEEL)
    # Low side link clears the opposite pad, which must contact the fixed handle.
    link( -0.256, d.GATE_X+0.002, -0.029, -0.025, 0.0005, 0.0035, STEEL)
    link( d.GATE_X-0.002, d.GATE_X+0.002, -0.029, -0.025, 0.0035, d.GATE_Z, STEEL)
    link( d.GATE_X-0.002, d.GATE_X+0.002, -0.029, -0.011,
         d.GATE_Z-d.GATE_T/2, d.GATE_Z+d.GATE_T/2, BLUE)
    return b.save()


BUILDERS = {"plunger": plunger, "shaft": shaft, "head": head, "cartridge": cartridge, "hood": hood, "tray": tray, "stand": stand}
