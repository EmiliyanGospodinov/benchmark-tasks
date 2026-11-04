"""USD rigid-body builder copied from A6 3a11129 (original A4 b987934). Import after Isaac Sim starts."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


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

