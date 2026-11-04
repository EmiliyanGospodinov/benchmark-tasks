"""USD bodies for the tipper truck: LDraw visual meshes with simplified box/cylinder colliders.

Import only after Isaac Sim has started (uses pxr). Each body's frame is the part frame
documented in ``design.py``. Colliders keep the mating surfaces, outer size and grasp
surfaces; studs, ribs, arches, printed details and the tow pin are visual only. The cab and
wheels are original approximations (no LDraw model exists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import design as d

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "assets" / "generated"

YELLOW, BLACK, AZURE, WHITE = (0.98, 0.78, 0.05), (0.08, 0.08, 0.09), (0.36, 0.71, 0.86), (0.95, 0.95, 0.95)
TYRE, HUB = (0.06, 0.06, 0.06), (0.98, 0.78, 0.05)


def _linear(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def _new_stage(path):
    stage = Usd.Stage.CreateNew(str(path)) if not path.exists() else Usd.Stage.Open(str(path))
    stage.GetRootLayer().Clear()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


class _Body:
    def __init__(self, name: str, mass: float, friction=(0.45, 0.35), com=None, stage=None, root="/Part",
                 offset=(0, 0, 0), iterations=(16, 4)):
        """A rigid body at prim ``root``; a new file ``name``.usda unless an open ``stage`` is given."""
        self.root = root
        if stage is None:
            GENERATED.mkdir(parents=True, exist_ok=True)
            self.path = GENERATED / f"{name}.usda"
            self.stage = _new_stage(self.path)
        else:
            self.path, self.stage = None, stage
        xf = UsdGeom.Xform.Define(self.stage, root)
        if stage is None:
            self.stage.SetDefaultPrim(xf.GetPrim())
        if any(offset):
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, offset)))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(mass))
        if com is not None:
            m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSolverPositionIterationCountAttr(int(iterations[0]))
        body.CreateSolverVelocityIterationCountAttr(int(iterations[1]))
        body.CreateMaxDepenetrationVelocityAttr(0.3)
        body.CreateLinearDampingAttr(0.05)
        body.CreateAngularDampingAttr(0.05)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.physics = UsdShade.Material.Define(self.stage, f"{root}/physics_material")
        pm = UsdPhysics.MaterialAPI.Apply(self.physics.GetPrim())
        pm.CreateStaticFrictionAttr(friction[0])
        pm.CreateDynamicFrictionAttr(friction[1])
        pm.CreateRestitutionAttr(0.0)
        self._materials: dict = {}
        self._n = 0
        self.boxes: list = []  # (lo, hi) of every collider box, for offline checks

    def material(self, color):
        key = tuple(round(c, 3) for c in color)
        if key not in self._materials:
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/look_{len(self._materials)}")
            sh = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("shader"))
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_linear(color)))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self._materials[key] = mat
        return self._materials[key]

    def mesh(self, npz: str, color, offset=(0, 0, 0)):
        data = np.load(ROOT / "assets" / npz)
        mesh = UsdGeom.Mesh.Define(self.stage, f"{self.root}/visual_{self._next()}")
        mesh.CreatePointsAttr((data["vertices"] + np.asarray(offset)).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(data["faces"]))
        mesh.CreateFaceVertexIndicesAttr(data["faces"].reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material(color))

    def _next(self):
        self._n += 1
        return self._n

    def _collide(self, prim, visible, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        props = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        props.CreateContactOffsetAttr(0.0006)
        props.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.physics, materialPurpose="physics")
        if visible:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.material(color))
        else:
            UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")

    def box(self, lo, hi, color=None):
        """Axis-aligned collider box between corners lo and hi; visible when a colour is given."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.boxes.append((lo, hi))
        if self.stage is None:
            return
        shape = UsdGeom.Cube.Define(self.stage, f"{self.root}/box_{self._next()}")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        shape.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        self._collide(shape.GetPrim(), color is not None, color)

    def cylinder_x(self, center, radius, length, color=None, collide=True):
        shape = UsdGeom.Cylinder.Define(self.stage, f"{self.root}/cyl_{self._next()}")
        shape.CreateAxisAttr("X")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(length))
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if collide:
            self._collide(shape.GetPrim(), color is not None, color)
        elif color is not None:
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.material(color))

    def capsule_x(self, center, radius, length):
        """Invisible exact capsule collider along x (PhysX approximates USD cylinders)."""
        shape = UsdGeom.Capsule.Define(self.stage, f"{self.root}/cap_{self._next()}")
        shape.CreateAxisAttr("X")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(length))
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        self._collide(shape.GetPrim(), False, None)

    def save(self) -> Path:
        self.stage.GetRootLayer().Save()
        return self.path


def chassis() -> Path:
    b = _Body("chassis", d.MASS["chassis"])
    b.mesh("15314.npz", WHITE)
    w, l, h = d.CHASSIS_SIZE
    b.box((-w / 2, -l / 2, -h), (w / 2, l / 2, 0))
    return b.save()


def axle(name: str) -> Path:
    """Two tyres on one axle; the axle turns in the chassis about x on a bearing joint.

    Tyres collide as spheres of the tyre radius (capsules with a negligible length): an exact
    round rolling contact, which PhysX's approximated cylinders did not give. Visuals are
    12 mm wide tyres with hubs.
    """
    b = _Body(name, d.MASS[name], friction=(1.0, 0.9))
    for side in (-1, 1):
        b.cylinder_x((side * d.TIRE_X, 0, 0), d.TIRE_RADIUS, d.TIRE_WIDTH, TYRE, collide=False)
        b.capsule_x((side * d.TIRE_X, 0, 0), d.TIRE_RADIUS, 0.0005)
        b.cylinder_x((side * (d.TIRE_X + d.TIRE_WIDTH / 2 + 0.0004), 0, 0), 0.008, 0.001, HUB, collide=False)
    return b.save()


def brick() -> Path:
    b = _Body("brick", d.MASS["brick"])
    b.mesh("3437.npz", AZURE, offset=(0, 0, d.BRICK_H / 2))
    b.box((-0.016, -0.016, -d.BRICK_H / 2), (0.016, 0.016, d.BRICK_H / 2))
    return b.save()


def cab() -> Path:
    """Original approximation of 15454pb06: cabin at the front, a 4 x 4.5 stud deck behind it.

    136 x 64 x 86.4 mm overall. Its skirts clear the base's wheel arches; it rests on the
    chassis deck. The brick of step 1 stands inside the open-bottomed cabin as the seat.
    """
    b = _Body("cab", d.MASS["cab"], com=(0, -0.025, 0.030))
    x0, x1, t = 0.032, 0.029, 0.003
    front, rear, back_of_cabin = -0.072, 0.064, -0.008
    top, sill, deck = 0.0864, 0.045, d.CAB_DECK_Z
    for s in (-1, 1):
        for y0, y1 in ((front, -0.0525), (-0.0115, 0.0115), (0.0525, rear)):  # skirts between arches
            b.box((min(s * x0, s * x1), y0, 0), (max(s * x0, s * x1), y1, 0.0096), YELLOW)
        b.box((min(s * x0, s * x1), front, 0.0096), (max(s * x0, s * x1), rear, deck), YELLOW)
        b.box((min(s * x0, s * x1), front, deck), (max(s * x0, s * x1), back_of_cabin, sill), YELLOW)
        for y in (front + 0.002, back_of_cabin - 0.005):  # window pillars
            b.box((min(s * x0, s * (x0 - 0.005)), y, sill), (max(s * x0, s * (x0 - 0.005)), y + 0.004, top), YELLOW)
    b.box((-x1, front, 0), (x1, front + t, 0.055), BLACK)  # grille
    b.box((-x0, front, top - 0.004), (x0, back_of_cabin, top), YELLOW)  # roof
    b.box((-x0, back_of_cabin - t, deck), (x0, back_of_cabin, top), YELLOW)  # headboard
    b.box((-x1, back_of_cabin, 0.0096), (x1, rear, deck), YELLOW)  # rear deck
    return b.save()


def hinge() -> Path:
    b = _Body("hinge", d.MASS["hinge"], iterations=(32, 8))  # light parts under a press: less contact sink
    b.mesh("65084.npz", BLACK)
    b.box((-0.016, -0.016, -d.PLATE_H), (0.016, 0.040, 0))  # plate
    for s in (-1, 1):  # knuckle ears
        b.box((min(s * 0.011, s * 0.016), 0.040, -d.PLATE_H), (max(s * 0.011, s * 0.016), 0.0556, 0.0096))
    ax = d.HINGE_AXIS_IN_PLATE
    b.box((-0.011, ax[1] - 0.0025, ax[2] - 0.0025), (0.011, ax[1] + 0.0025, ax[2] + 0.0025))  # pin bar
    return b.save()


def bed() -> Path:
    """Tipper bed 14094. Origin on the hinge axis; it extends 64 mm forward (-y). The rear is open between
    the side walls down to the floor (LDraw mesh: nothing above 8.9 mm behind y = 0 except the side walls);
    the load leaves through it. Earlier colliders added a rear lip and two rear posts the part does not have
    (they held the payload in at 55 deg in bed-021)."""
    b = _Body("bed", d.MASS["bed"], com=(0, -0.030, 0.012), iterations=(32, 8))
    b.mesh("14094.npz", AZURE)
    floor = (0.0026, 0.0076)
    # Cutouts leave 2.5 mm around the hinge plate's knuckle ears (|x| 11-16 mm, y -8..+7.6 mm here).
    b.box((-0.032, -0.064, floor[0]), (0.032, -0.0105, floor[1]))
    b.box((-0.0095, -0.0105, floor[0]), (0.0095, -0.0045, floor[1]))
    for s in (-1, 1):
        b.box((min(s * 0.0175, s * 0.032), -0.0105, floor[0]), (max(s * 0.0175, s * 0.032), 0.0, floor[1]))
        b.box((min(s * 0.028, s * 0.032), -0.064, 0.0), (max(s * 0.028, s * 0.032), 0.008, 0.0364))  # sides
        # Snap clip: a bridge resting on the hinge pin with two prongs either side of it, which
        # locate the bed along y (1 mm play) as the real C-shaped clip does.
        lo_x, hi_x = min(s * 0.003, s * 0.010), max(s * 0.003, s * 0.010)
        b.box((lo_x, -0.0050, 0.0025), (hi_x, 0.0050, 0.0085))
        for y0, y1 in ((-0.0050, -0.0035), (0.0035, 0.0050)):
            b.box((lo_x, y0, -0.0015), (hi_x, y1, 0.0025))
    b.box((-0.032, -0.064, 0.0), (0.032, -0.060, 0.0364))  # front wall
    return b.save()


BUILDERS = {"chassis": chassis, "axle_front": lambda: axle("axle_front"), "axle_rear": lambda: axle("axle_rear"),
            "brick": brick, "cab": cab, "hinge": hinge, "bed": bed}
