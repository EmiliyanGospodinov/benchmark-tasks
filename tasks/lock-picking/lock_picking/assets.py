"""USD rigid bodies of the clear pin-tumbler trainer (primitives and convex prisms, PhysX contact offsets).

Body builder adapted from our A26 ratchet-lowering task (yambition-a26-ratchet-lowering, ratchet_lowering/assets.py,
MIT). Import after Isaac Sim starts. Colliders that are not the visible shape are authored with purpose "guide"
(not rendered); see-through parts use a UsdPreviewSurface opacity.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import design as d

GENERATED = Path(__file__).resolve().parents[1] / "assets" / "generated"

CLEAR = (0.70, 0.85, 0.95)
BRASS = (0.85, 0.68, 0.25)
DRIVER_COLORS = ((0.85, 0.15, 0.15), (0.15, 0.65, 0.20), (0.15, 0.35, 0.90))
STEEL, DARK, ORANGE, WHITE, GREEN = (0.62, 0.64, 0.67), (0.22, 0.24, 0.27), (0.95, 0.45, 0.05), (0.95, 0.95, 0.95), (0.1, 0.8, 0.2)


def _linear(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


class Body:
    def __init__(self, name, mass, friction, kinematic=False, com=None):
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
        body.CreateSolverVelocityIterationCountAttr(4)
        body.CreateMaxDepenetrationVelocityAttr(0.2)
        self.physics = UsdShade.Material.Define(self.stage, "/Part/physics_material")
        pm = UsdPhysics.MaterialAPI.Apply(self.physics.GetPrim())
        pm.CreateStaticFrictionAttr(friction[0])
        pm.CreateDynamicFrictionAttr(friction[1])
        pm.CreateRestitutionAttr(0.0)
        self._looks, self._n = {}, 0

    def _look(self, color, opacity):
        key = (color, opacity)
        if key not in self._looks:
            mat = UsdShade.Material.Define(self.stage, f"/Part/look_{len(self._looks)}")
            sh = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("shader"))
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_linear(color)))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            if opacity < 1.0:
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self._looks[key] = mat
        return self._looks[key]

    def _finish(self, prim, color, collide, visible, opacity=1.0, mesh=False):
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            if mesh:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            props = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            props.CreateContactOffsetAttr(0.001)
            props.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.physics, materialPurpose="physics")
        if visible:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._look(color, opacity))
        else:
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)

    def box(self, lo, hi, color=STEEL, collide=True, visible=True, opacity=1.0, rot_x=0.0):
        """Axis-aligned box from corner lo to hi (body frame); optionally turned rot_x about +x at its centre."""
        self._n += 1
        shape = UsdGeom.Cube.Define(self.stage, f"/Part/box_{self._n}")
        shape.CreateSizeAttr(1.0)
        c = [(a + b) / 2 for a, b in zip(lo, hi)]
        s = [abs(b - a) for a, b in zip(lo, hi)]
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        if rot_x:
            shape.AddRotateXOp().Set(math.degrees(rot_x))
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, s)))
        self._finish(shape.GetPrim(), color, collide, visible, opacity)

    def cylinder_x(self, x0, x1, radius, color, y=0.0, z=0.0, collide=True, visible=True, opacity=1.0):
        self._n += 1
        shape = UsdGeom.Cylinder.Define(self.stage, f"/Part/cyl_{self._n}")
        shape.CreateAxisAttr("X")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(x1 - x0))
        shape.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, float(y), float(z)))
        self._finish(shape.GetPrim(), color, collide, visible, opacity)

    def prism_x(self, yz_points, x0, x1, color=STEEL, collide=True, visible=False):
        """Convex prism: a convex polygon in the y-z plane (counter-clockwise) extruded along x."""
        self._n += 1
        mesh = UsdGeom.Mesh.Define(self.stage, f"/Part/prism_{self._n}")
        n = len(yz_points)
        pts = [Gf.Vec3f(float(x0), float(y), float(z)) for y, z in yz_points] + \
              [Gf.Vec3f(float(x1), float(y), float(z)) for y, z in yz_points]
        counts, idx = [n, n], list(range(n))[::-1] + list(range(n, 2 * n))
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            idx += [i, j, n + j, n + i]
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateSubdivisionSchemeAttr("none")
        self._finish(mesh.GetPrim(), color, collide, visible, mesh=True)

    def save(self) -> Path:
        self.stage.GetRootLayer().Save()
        return self.path


def _polar(r, phi):
    """(y, z) at radius r and angle phi from +z toward +y."""
    return (r * math.sin(phi), r * math.cos(phi))


def _arc_piece(r_out, phi0, phi1, r_in, n=5):
    pts = [_polar(r_out, phi0 + (phi1 - phi0) * k / n) for k in range(n + 1)]
    pts += [_polar(r_in, phi1), _polar(r_in, phi0)]
    return pts


def build_housing() -> Path:
    """Fixed (kinematic) housing on its pedestal; frame on the plug axis at the keyway face."""
    h = Body("housing", 2.0, d.FRICTION["housing"], kinematic=True)
    zb = -(d.AXIS_Z - d.TOP)
    h.box((0.0, -0.032, zb), (d.HOUSING_L, 0.032, -0.032), DARK)  # pedestal, clamped to the table
    # cut-away housing: only the chamber walls are drawn (thin clear-grey fins between the stacks)
    for px in (d.PIN_X[0] - 0.0075, d.PIN_X[1] - 0.0075, d.PIN_X[2] - 0.0075, d.PIN_X[2] + 0.0075):
        h.box((px - 0.0008, -0.010, d.R_BORE), (px + 0.0008, 0.010, 0.052), CLEAR, collide=False, opacity=0.35)
    for i, px in enumerate(d.PIN_X):  # chamber roof: blocks a key pin pushed past the shear line
        for s in (-1, 1):
            lo, hi = sorted((s * (d.PIN_W / 2 + 0.0003), s * 0.010))
            h.box((px - d.SLICE, lo, d.R_BORE), (px + d.SLICE, hi, 0.034), visible=False)
    # bore wall where the key pins face at the open mark (keeps them in the plug when it is turned)
    for px in d.PIN_X:
        h.box((px - d.SLICE, -0.030, -0.006), (px + d.SLICE, -d.R_BORE, 0.006), visible=False)
    # face marks: white tick at the start pointer direction, green tick at the open mark (90 degrees)
    for deg, color in ((0.0, WHITE), (d.THETA_OPEN_DEG, GREEN)):
        u = _dir_yz(math.radians(deg))
        c = 0.028 * u
        h.box((-0.0012, c[0] - 0.002, c[1] - 0.002), (-0.0002, c[0] + 0.002, c[1] + 0.002), color, collide=False)
    return h.save()


def _dir_yz(theta):
    return d.lever_dir(theta)


def build_plug() -> Path:
    p = Body("plug", d.MASS["plug"], d.FRICTION["plug"], com=(d.PLUG_L / 2, 0.0, 0.0))
    p.cylinder_x(0.0, d.PLUG_L, d.R - 0.0002, CLEAR, collide=False, opacity=0.12)
    # keyway (pick guide): floor, side walls, blind end, roof between the pin holes
    hw, fl, rf, dep = d.KEYWAY_HALF_W, d.KEYWAY_FLOOR, d.KEYWAY_ROOF, d.KEYWAY_DEPTH
    p.box((0.0, -0.008, fl - 0.003), (dep, 0.008, fl), visible=False)
    for s in (-1, 1):
        lo, hi = sorted((s * hw, s * (hw + 0.004)))
        p.box((0.0, lo, fl), (dep, hi, rf), visible=False)
    p.box((dep, -hw, fl), (dep + 0.005, hw, rf), visible=False)
    edges = [0.0] + [e for px in d.PIN_X for e in (px - d.HOLE_X / 2, px + d.HOLE_X / 2)] + [dep]
    for a, b in zip(edges[::2], edges[1::2]):
        p.box((a, -hw, rf), (b, hw, rf + 0.004), visible=False)
    p.box((0.0, -hw, fl), (0.001, hw, rf), DARK, collide=False, opacity=0.6)  # keyway mouth (visual)
    # driver-side surface under each driver: binding wall, shear-line arc, open detent notch
    r_in = d.R - 0.004
    notch = math.asin(d.NOTCH_HALF / d.R)
    for i, px in enumerate(d.PIN_X):
        x0, x1 = px - d.SLICE, px + d.SLICE
        yw = d.PIN_W / 2 + d.GAP[i]
        ze = math.sqrt(d.R ** 2 - yw ** 2)
        phi_e = math.atan2(yw, ze)
        phi_w = phi_e + math.radians(12)
        wall = [(yw, 0.009)] + [_polar(d.R, phi_e + (phi_w - phi_e) * k / 4) for k in range(5)] + [_polar(r_in, phi_w)]
        p.prism_x(wall, x0, x1)
        a, end = phi_w, math.pi / 2 - notch
        while a < end - 1e-6:
            b = min(a + math.radians(15), end)
            p.prism_x(_arc_piece(d.R, a, b, r_in), x0, x1)
            a = b
        p.prism_x(_arc_piece(d.R - d.NOTCH_DEPTH, math.pi / 2 - notch, math.pi / 2 + notch, r_in), x0, x1)
        p.prism_x(_arc_piece(d.R, math.pi / 2 + notch, math.pi / 2 + notch + math.radians(15), r_in), x0, x1)
    # zero pointer on the face (the wrench's unflexed direction)
    u = d.lever_dir(0.0)
    c = 0.017 * u
    p.box((-0.0010, c[0] - 0.0012, c[1] - 0.0012), (0.0, c[0] + 0.0012, c[1] + 0.0012), ORANGE, collide=False)
    return p.save()


def build_lever() -> Path:
    lv = Body("lever", d.MASS["lever"], d.FRICTION["lever"])
    a = math.radians(d.LEVER_DIR0_DEG)
    u = d.lever_dir(0.0)
    mid = u * (d.LEVER_R0 + d.LEVER_L) / 2
    length = d.LEVER_L - d.LEVER_R0 + 0.006
    # bar along the lever direction: a box along y turned about x so that +y points along u
    turn = math.atan2(u[1], u[0])
    lv.box((-0.003, mid[0] - length / 2, mid[1] - 0.005), (0.003, mid[0] + length / 2, mid[1] + 0.005),
           (0.25, 0.25, 0.28), rot_x=turn)
    tip = u * d.LEVER_L
    lv.cylinder_x(-0.003 - 0.006 - d.SLEEVE_L - 0.002, -0.003, 0.003, STEEL, y=tip[0], z=tip[1], collide=False)
    del a
    return lv.save()


def build_sleeve() -> Path:
    s = Body("sleeve", d.MASS["sleeve"], d.FRICTION["sleeve"], com=(0.0, 0.0, 0.0))
    s.cylinder_x(-d.SLEEVE_L / 2, d.SLEEVE_L / 2, d.SLEEVE_R, ORANGE)
    return s.save()


def build_pin(kind: str, i: int) -> Path:
    length = d.KEY_L if kind == "key" else d.DRIVER_L
    color = BRASS if kind == "key" else DRIVER_COLORS[i]
    b = Body(f"{kind}{i}", d.MASS[kind], d.FRICTION["pin"], com=(0.0, 0.0, 0.0))
    w = d.PIN_W / 2
    b.box((-w, -w, -length / 2), (w, w, length / 2), color)
    if kind == "driver":  # indicator cap (visual)
        cx, cy, cz = d.CAP
        b.box((-cx / 2, -cy / 2, length / 2), (cx / 2, cy / 2, length / 2 + cz), color, collide=False)
    return b.save()


def build_pick(kinematic=False) -> Path:
    name = "pick_diag" if kinematic else "pick"
    k = Body(name, d.MASS["pick"], d.FRICTION["pick"], kinematic=kinematic)
    L, W, H = d.PICK_SHAFT
    k.box((-L, -W / 2, -H / 2), (0.0, W / 2, H / 2), STEEL)  # shaft, frame at its tip centre
    hx, hw, hh = d.PICK_HOOK
    k.box((-hx, -hw / 2, H / 2), (0.0, hw / 2, H / 2 + hh), STEEL)  # upturned tooth
    k.box((-L, -W / 2, -H / 2), (-L + W, d.PICK_LEG_Y + d.POST[1] / 2, H / 2), STEEL)  # dog-leg
    px, py, pz = d.POST
    k.box((-L - px / 2 + W / 2, d.PICK_LEG_Y - py / 2, -H / 2), (-L + px / 2 + W / 2, d.PICK_LEG_Y + py / 2, -H / 2 + pz),
          (0.10, 0.10, 0.12))  # upright handle post
    return k.save()


def post_center_local(z_above_bottom: float) -> tuple[float, float, float]:
    """Pick-frame point on the post axis at a height above the shaft bottom."""
    L, W, H = d.PICK_SHAFT
    return (-L + W / 2, d.PICK_LEG_Y, -H / 2 + z_above_bottom)


BUILDERS = {"housing": build_housing, "plug": build_plug, "lever": build_lever, "sleeve": build_sleeve,
            "pick": build_pick, **{f"key{i}": (lambda i=i: build_pin("key", i)) for i in range(3)},
            **{f"driver{i}": (lambda i=i: build_pin("driver", i)) for i in range(3)}}
