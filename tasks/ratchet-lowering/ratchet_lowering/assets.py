"""USD rigid bodies of the ratchet trainer (original simplified geometry). Import after Isaac Sim starts.

The body builder is adapted from our A5 inspection-mount task (yambition-a5-inspection-mount @ b48bac2,
mount_aim/assets.py, MIT): primitives with PhysX contact offsets; added here: pitched boxes, cylinders
along y, and convex-hull wedge meshes for the ratchet teeth.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import design as d

GENERATED = Path(__file__).resolve().parents[1] / "assets" / "generated"
STEEL, DARK, ORANGE, BLUE, WOOD, WHITE = ((0.62, 0.64, 0.67), (0.22, 0.24, 0.27), (0.95, 0.50, 0.10),
                                          (0.15, 0.35, 0.80), (0.70, 0.55, 0.38), (0.92, 0.93, 0.95))
GREEN = (0.10, 0.65, 0.20)


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

    def _finish(self, prim, color, collide=True, mesh=False):
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            if mesh:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            props = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            props.CreateContactOffsetAttr(0.001)
            props.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.physics, materialPurpose="physics")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._look(color))

    def box(self, center, size, color, pitch=0.0, collide=True):
        """Box; ``pitch`` rotates it about +y (radians, same sense as the rotor joint)."""
        self._n += 1
        shape = UsdGeom.Cube.Define(self.stage, f"/Part/box_{self._n}")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if pitch:
            shape.AddRotateYOp().Set(math.degrees(pitch))
        shape.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        self._finish(shape.GetPrim(), color, collide)

    def cylinder_y(self, y0, y1, radius, color, x=0.0, z=0.0, collide=True):
        self._n += 1
        shape = UsdGeom.Cylinder.Define(self.stage, f"/Part/cyl_{self._n}")
        shape.CreateAxisAttr("Y")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(y1 - y0))
        shape.AddTranslateOp().Set(Gf.Vec3d(float(x), (y0 + y1) / 2, float(z)))
        self._finish(shape.GetPrim(), color, collide)

    def prism_y(self, xz_points, y0, y1, color, collide=True):
        """Convex prism: a polygon in the x-z plane extruded along y (convex-hull collider)."""
        self._n += 1
        mesh = UsdGeom.Mesh.Define(self.stage, f"/Part/prism_{self._n}")
        n = len(xz_points)
        pts = [Gf.Vec3f(float(x), float(y0), float(z)) for x, z in xz_points] + \
              [Gf.Vec3f(float(x), float(y1), float(z)) for x, z in xz_points]
        counts, idx = [n, n], list(range(n))[::-1] + list(range(n, 2 * n))
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            idx += [i, j, n + j, n + i]
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateSubdivisionSchemeAttr("none")
        self._finish(mesh.GetPrim(), color, collide, mesh=True)

    def save(self) -> Path:
        self.stage.GetRootLayer().Save()
        return self.path


def frame() -> Path:
    """Fixed trainer frame (kinematic, clamped to the table): base, two shaft uprights, pawl bracket, rest."""
    b = _Body("frame", 3.0, d.FRICTION["frame"], kinematic=True)
    zs = d.SHAFT[2] - d.TOP
    b.box((0.03, -0.01, 0.004), (0.22, 0.30, 0.008), DARK)
    for y in (-0.050, 0.050):  # uprights between lever and wheel / crank; the shaft is visual where it passes
        b.box((0.0, y, (zs + 0.012) / 2), (0.030, 0.010, zs + 0.012), STEEL)
    px, pz = d.PAWL_PIVOT
    b.box((px, d.PAWL_Y - 0.012, (zs + pz) / 2), (0.012, 0.006, zs + pz), STEEL)  # pawl bracket
    b.cylinder_y(d.PAWL_Y - 0.012, d.PAWL_Y + 0.006, 0.003, DARK, x=px, z=zs + pz, collide=False)  # pin (visual)
    # receiving rest: a pad whose top is the weight's lower face at THETA_REST, on a pedestal
    phi = -math.radians(d.THETA_REST_DEG)
    r = d.rot_y(phi)
    c = r @ [d.LEVER_L, 0.0, -d.WEIGHT[2] / 2 - 0.010]
    b.box((c[0], 0.0, zs + c[2]), (0.050, 0.050, 0.020), WOOD, pitch=phi)
    top = zs + c[2]
    b.box((c[0] + 0.005, 0.0, top / 2 - 0.004), (0.030, 0.040, top - 0.008), WOOD)
    return b.save()


def rotor() -> Path:
    """Shaft, ratchet wheel with 24 sawtooth teeth, load lever with its weight, crank arm with pin."""
    b = _Body("rotor", d.MASS["rotor"], d.FRICTION["rotor"], com=d.ROTOR_COM)
    b.cylinder_y(-0.100, 0.100, 0.006, STEEL, collide=False)  # shaft (runs in the uprights' bearings)
    y0, y1 = d.WHEEL_Y - d.WHEEL_T / 2, d.WHEEL_Y + d.WHEEL_T / 2
    b.cylinder_y(y0, y1, d.ROOT_R, DARK)
    p = math.radians(d.TOOTH_DEG)
    g0 = math.radians(d.tooth_phase_deg())
    rr = d.ROOT_R - 0.001  # overlap into the disc
    for k in range(d.TEETH):
        g = g0 + k * p  # steep face (toward decreasing angle); ramp down to the next root
        pts = [(rr * math.cos(g), rr * math.sin(g)), (d.TIP_R * math.cos(g), d.TIP_R * math.sin(g)),
               (rr * math.cos(g + p), rr * math.sin(g + p))]
        b.prism_y(pts, y0, y1, STEEL)
    L = d.LEVER_L - d.WEIGHT[0] / 2
    b.box((L / 2, 0.0, 0.0), (L, d.LEVER_W, d.LEVER_T), STEEL)
    b.box((d.LEVER_L, 0.0, 0.0), tuple(d.WEIGHT), BLUE)
    a = math.radians(d.CRANK_LOCAL_DEG)
    b.box((d.CRANK_R / 2 * math.cos(a), d.CRANK_Y, d.CRANK_R / 2 * math.sin(a)), (d.CRANK_R + 0.016, 0.008, 0.014),
          STEEL, pitch=-a)
    b.cylinder_y(d.CRANK_Y + 0.004, d.CRANK_Y + 0.006, 0.006, DARK, x=d.CRANK_R * math.cos(a),
                 z=d.CRANK_R * math.sin(a), collide=False)
    return b.save()


def handle() -> Path:
    """Crank handle sleeve: spins freely on the crank pin (its own revolute joint)."""
    b = _Body("handle", d.MASS["handle"], d.FRICTION["handle"], com=(0, 0, 0))
    b.cylinder_y(-d.HANDLE_L / 2, d.HANDLE_L / 2, d.HANDLE_R, ORANGE)
    return b.save()


def pawl() -> Path:
    """Pawl on its pivot: link, nose (engages the steep tooth faces) and a thumb tab beyond the pivot."""
    b = _Body("pawl", d.MASS["pawl"], d.FRICTION["pawl"], com=(0.0, 0.0, 0.0))
    nose = d.PAWL_NOSE - d.PAWL_PIVOT
    u = nose / math.hypot(*nose)
    a = math.atan2(-u[1], u[0])  # pitch (about +y) that lays a box's +x along the link direction
    length = math.hypot(*nose) - d.NOSE[0] / 2
    mid = u * length / 2
    b.box((mid[0], 0.0, mid[1]), (length, d.PAWL_T, 0.006), ORANGE, pitch=a)
    nc = nose - u * d.NOSE[0] / 2
    b.box((nc[0], 0.0, nc[1]), (d.NOSE[0], d.PAWL_T, d.NOSE[1]), ORANGE, pitch=a)
    tab = d.TAB_END - d.PAWL_PIVOT
    b.box((tab[0] / 2, 0.0, tab[1] / 2), (math.hypot(*tab), d.PAWL_T, 0.005), ORANGE,
          pitch=math.atan2(-tab[1], tab[0]))
    b.box((tab[0], 0.0, tab[1]), d.TAB, ORANGE)
    b.cylinder_y(-d.PAWL_T / 2 - 0.001, d.PAWL_T / 2 + 0.001, 0.006, DARK, collide=False)  # hub
    return b.save()


BUILDERS = {"frame": frame, "rotor": rotor, "handle": handle, "pawl": pawl}
