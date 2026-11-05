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
        guide_body = True
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

