#!/usr/bin/env python3
"""Render randomized bulldozer foreground captures from Blender.

Run this script with Blender, not regular Python:

  blender --background blender/dozer.blend --python scripts/render_blender_captures.py -- \
    --output-dir data/captures/blender_generated --train-count 300 --test-count 60

It writes capture pairs compatible with scripts/rebuild_dozer_dataset.py:

  data/captures/blender_generated/train/image_000000_img.png
  data/captures/blender_generated/train/image_000000_layer.png

The image render uses the model's original materials over a transparent
background. The layer render temporarily replaces visible model materials with
solid red emission, producing a mask that the compositor can threshold.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector


DEFAULT_ROOTS = ("Sketchfab Model", "Sketchfab Model.001", "Sketchfab Model.002")


@dataclass(frozen=True)
class ModelRoot:
    obj: bpy.types.Object
    center: Vector
    size: Vector
    radius: float


def parse_args() -> argparse.Namespace:
    args = sys.argv
    if "--" in args:
        args = args[args.index("--") + 1 :]
    else:
        args = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/captures/blender_generated"))
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=60)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--engine", choices=["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"], default="BLENDER_EEVEE")
    return parser.parse_args(args)


def root_meshes(root: bpy.types.Object) -> list[bpy.types.Object]:
    return [obj for obj in [root, *root.children_recursive] if obj.type == "MESH"]


def compute_root(root: bpy.types.Object) -> ModelRoot:
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in root_meshes(root):
        found = True
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
    if not found:
        raise ValueError(f"Root {root.name!r} has no mesh descendants")
    size = maxs - mins
    center = (mins + maxs) * 0.5
    return ModelRoot(obj=root, center=center, size=size, radius=max(size.length / 2, 1.0))


def ensure_camera() -> bpy.types.Object:
    camera = bpy.context.scene.camera
    if camera is not None:
        return camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    return camera


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    engine = args.engine
    if engine not in available_engines and engine == "BLENDER_EEVEE_NEXT" and "BLENDER_EEVEE" in available_engines:
        engine = "BLENDER_EEVEE"
    scene.render.engine = engine
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    if engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples


def set_root_visibility(roots: list[ModelRoot], active: ModelRoot) -> None:
    for root in roots:
        hide = root.obj.name != active.obj.name
        root.obj.hide_viewport = hide
        root.obj.hide_render = hide
        for child in root.obj.children_recursive:
            child.hide_viewport = hide
            child.hide_render = hide


def randomize_camera(camera: bpy.types.Object, model: ModelRoot, rng: random.Random) -> None:
    azimuth = rng.uniform(0, math.tau)
    elevation = math.radians(rng.uniform(8, 38))
    distance = model.radius * rng.uniform(2.2, 3.5)
    target = model.center + Vector(
        (
            rng.uniform(-0.08, 0.08) * model.size.x,
            rng.uniform(-0.08, 0.08) * model.size.y,
            rng.uniform(-0.05, 0.12) * model.size.z,
        )
    )
    camera.location = Vector(
        (
            target.x + distance * math.cos(elevation) * math.cos(azimuth),
            target.y + distance * math.cos(elevation) * math.sin(azimuth),
            target.z + distance * math.sin(elevation),
        )
    )
    look_at(camera, target)
    camera.data.lens = rng.uniform(38, 75)
    camera.data.dof.use_dof = rng.random() < 0.25
    camera.data.dof.focus_object = model.obj
    camera.data.dof.aperture_fstop = rng.uniform(5.6, 11.0)


def ensure_area_light(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj and obj.type == "LIGHT":
        return obj
    bpy.ops.object.light_add(type="AREA")
    obj = bpy.context.object
    obj.name = name
    return obj


def randomize_lighting(model: ModelRoot, rng: random.Random) -> None:
    key = ensure_area_light("Synthetic_Key_Area")
    fill = ensure_area_light("Synthetic_Fill_Area")
    for light, strength, size in (
        (key, rng.uniform(450, 1600), rng.uniform(3.0, 7.0)),
        (fill, rng.uniform(40, 450), rng.uniform(5.0, 12.0)),
    ):
        angle = rng.uniform(0, math.tau)
        height = rng.uniform(0.8, 2.3) * model.radius
        distance = rng.uniform(1.8, 3.8) * model.radius
        light.location = Vector(
            (
                model.center.x + math.cos(angle) * distance,
                model.center.y + math.sin(angle) * distance,
                model.center.z + height,
            )
        )
        look_at(light, model.center)
        light.data.energy = strength
        light.data.size = size
        light.hide_render = False
        light.hide_viewport = False

    world = bpy.context.scene.world or bpy.data.worlds.new("Synthetic World")
    bpy.context.scene.world = world
    world.color = (rng.uniform(0.0, 0.08), rng.uniform(0.0, 0.08), rng.uniform(0.0, 0.08))


def make_mask_material() -> bpy.types.Material:
    material = bpy.data.materials.get("Synthetic_Mask_Red")
    if material is None:
        material = bpy.data.materials.new("Synthetic_Mask_Red")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1, 0, 0, 1)
        emission.inputs["Strength"].default_value = 1
        output = nodes.new(type="ShaderNodeOutputMaterial")
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def render_still(path: Path, transparent: bool) -> None:
    scene = bpy.context.scene
    scene.render.film_transparent = transparent
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_pair(path_base: Path, model: ModelRoot, mask_material: bpy.types.Material) -> None:
    image_path = path_base.with_name(path_base.name + "_img.png")
    layer_path = path_base.with_name(path_base.name + "_layer.png")

    render_still(image_path, transparent=True)

    original_slots: list[tuple[bpy.types.Object, list[bpy.types.Material | None]]] = []
    for obj in root_meshes(model.obj):
        original_slots.append((obj, [slot.material for slot in obj.material_slots]))
        if not obj.material_slots:
            obj.data.materials.append(mask_material)
        for slot in obj.material_slots:
            slot.material = mask_material

    try:
        render_still(layer_path, transparent=False)
    finally:
        for obj, materials in original_slots:
            while len(obj.material_slots) > len(materials):
                obj.data.materials.pop(index=len(obj.material_slots) - 1)
            for slot, material in zip(obj.material_slots, materials):
                slot.material = material


def render_split(
    *,
    split: str,
    count: int,
    output_dir: Path,
    roots: list[ModelRoot],
    camera: bpy.types.Object,
    mask_material: bpy.types.Material,
    rng: random.Random,
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        model = rng.choice(roots)
        set_root_visibility(roots, model)
        randomize_camera(camera, model, rng)
        randomize_lighting(model, rng)
        render_pair(split_dir / f"image_{index:06d}", model, mask_material)
        print(f"{split}: rendered {index + 1}/{count} from {model.obj.name}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    configure_render(args)

    roots: list[ModelRoot] = []
    for name in args.roots:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise ValueError(f"Could not find model root {name!r}")
        roots.append(compute_root(obj))

    camera = ensure_camera()
    mask_material = make_mask_material()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    render_split(
        split="train",
        count=args.train_count,
        output_dir=args.output_dir,
        roots=roots,
        camera=camera,
        mask_material=mask_material,
        rng=rng,
    )
    render_split(
        split="test",
        count=args.test_count,
        output_dir=args.output_dir,
        roots=roots,
        camera=camera,
        mask_material=mask_material,
        rng=rng,
    )


if __name__ == "__main__":
    main()
