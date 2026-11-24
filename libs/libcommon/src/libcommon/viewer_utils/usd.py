# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

"""Convert OpenUSD files (.usd, .usda, .usdc and .usdz) to trimesh scenes.

trimesh doesn't read USD: the thumbnail of a USD cell is rendered from the trimesh scene it's converted to. The
viewer renders the USD file itself.

Only the geometry and the base colors are converted: the visible meshes of the stage, with their transforms,
their instances, and the constant values of their UsdPreviewSurface materials (or their displayColor). The
textures, the MDL materials, the point instancers, the lights and the cameras are ignored.
"""

import logging
import os
import posixpath
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    import trimesh
    from huggingface_hub import HfFileSystem

logger = logging.getLogger(__name__)

USD_FILE_TYPES = {"usd", "usda", "usdc", "usdz"}
# The color of the meshes without a material nor a displayColor
USD_DEFAULT_COLOR = (0.7, 0.7, 0.7)
# A USD file can compose other USD files (sublayers, references, payloads): a SimReady asset is a small root
# file with its geometry in a payload. These limits bound the downloads of the files of a single cell.
USD_MAX_LAYERS = 64
# A few kilobytes of references can compose an exponential number of prims: these limits bound the conversion
USD_MAX_PRIMS = 1_000_000
USD_MAX_TRIANGLES = 20_000_000


def download_usd_layers(
    fs: "HfFileSystem", repo_prefix: str, root_path: str, local_dir: str, max_total_size: int
) -> str:
    """Download a USD file of a Hub repository, and the USD files it composes, and return the local root file.

    The repository is given by its prefix ("hf://datasets/{dataset}@{revision}/"), and the file by its path in
    it. The files are downloaded at the same place relative to the repository root, so that the relative paths
    of the composition arcs resolve to them. The files out of the repository, the textures and the MDL
    materials are not downloaded. It raises a ValueError if there are too many files, or if they are too large
    in total.
    """
    path = repo_prefix + root_path
    to_download, downloaded, total_size = [root_path], set[str](), 0
    while to_download:
        repo_path = to_download.pop()
        if repo_path in downloaded:
            continue
        if len(downloaded) >= USD_MAX_LAYERS:
            raise ValueError(f"The USD file {path} composes more than {USD_MAX_LAYERS} files.")
        downloaded.add(repo_path)
        try:
            total_size += fs.size(repo_prefix + repo_path)
        except FileNotFoundError:
            # the stage is composed without the missing file, as USD does for any missing layer
            logger.info(f"Skipping the missing USD file {repo_path} composed by {path}.")
            continue
        if total_size > max_total_size:
            raise ValueError(f"The USD file {path} and the files it composes are larger than {max_total_size} bytes.")
        local_path = os.path.join(local_dir, repo_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        fs.get_file(repo_prefix + repo_path, local_path)
        to_download.extend(_composed_usd_paths(local_path, repo_path))
    return os.path.join(local_dir, root_path)


def _composed_usd_paths(local_path: str, repo_path: str) -> list[str]:
    """Return the paths in the repository of the USD files that a USD file composes."""
    from pxr import Sdf

    # a .usdz package contains the files it composes
    if local_path.lower().endswith(".usdz"):
        return []
    layer = Sdf.Layer.FindOrOpen(local_path)
    if layer is None:
        raise ValueError(f"The USD file {repo_path} cannot be opened.")
    composed_paths = []
    for asset_path in layer.GetCompositionAssetDependencies():
        # an absolute path, or a URL, is not a file of the repository
        if "://" in asset_path or asset_path.startswith("/") or not asset_path:
            continue
        composed_path = posixpath.normpath(posixpath.join(posixpath.dirname(repo_path), asset_path))
        if composed_path.startswith("../") or composed_path.rsplit(".", 1)[-1].lower() not in USD_FILE_TYPES:
            continue
        composed_paths.append(composed_path)
    return composed_paths


def load_usd_scene(path: str, local_dir: str) -> "trimesh.Scene":
    """Load a local USD file as a trimesh scene: Y up and in meters, like glTF.

    The files it composes must be in local_dir, where the files of the cell are: USD resolves the absolute and
    relative paths of the composition arcs on the local disk, where other files must not be read from.
    """
    import trimesh
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise ValueError("The USD file cannot be opened.")
    local_root = os.path.join(os.path.realpath(local_dir), "")
    for layer in stage.GetUsedLayers():
        # the layers of a .usdz package are "package.usdz[layer.usd]", and the anonymous layers have no path
        layer_path = layer.realPath.split("[", 1)[0]
        if layer_path and not os.path.realpath(layer_path).startswith(local_root):
            raise ValueError("The USD file composes a file out of the dataset.")
    time = Usd.TimeCode.EarliestTime()
    xform_cache = UsdGeom.XformCache(time)
    root_transform = np.diag([UsdGeom.GetStageMetersPerUnit(stage)] * 3 + [1.0])
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        # a rotation of -90 degrees around X: Z up becomes Y up
        root_transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ root_transform

    scene = trimesh.Scene()
    # the meshes of the instances are the same prims of a prototype: they are converted once, and added to the
    # scene with the transform of each instance. An instance can inherit another material from its ancestors.
    meshes_of_prims: dict[str, list[trimesh.Trimesh]] = {}
    total_triangles = 0
    for prim_count, prim in enumerate(stage.Traverse(Usd.TraverseInstanceProxies())):
        if prim_count >= USD_MAX_PRIMS:
            raise ValueError(f"The USD file contains more than {USD_MAX_PRIMS} prims.")
        if not prim.IsA(UsdGeom.Mesh) or not _is_rendered(UsdGeom.Imageable(prim), time):
            continue
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        mesh_key = f"{_prototype_path(prim)}@{_prototype_path(material.GetPrim()) if material else ''}"
        if mesh_key not in meshes_of_prims:
            meshes_of_prims[mesh_key] = _convert_mesh(UsdGeom.Mesh(prim), time)
        total_triangles += sum(len(mesh.faces) for mesh in meshes_of_prims[mesh_key])
        if total_triangles > USD_MAX_TRIANGLES:
            raise ValueError(f"The USD file contains more than {USD_MAX_TRIANGLES} triangles.")
        # USD matrices transform row vectors: the transpose is the matrix of the column vectors of trimesh
        transform = root_transform @ np.array(xform_cache.GetLocalToWorldTransform(prim)).T
        for index, mesh in enumerate(meshes_of_prims[mesh_key]):
            geom_name, node_name = f"{mesh_key}#{index}", f"{prim.GetPath()}#{index}"
            if geom_name in scene.geometry:
                # add_geometry would add a copy of the geometry: the instances share it in the scene graph
                scene.graph.update(frame_to=node_name, matrix=transform, geometry=geom_name)  # type: ignore[no-untyped-call]
            else:
                scene.add_geometry(mesh, geom_name=geom_name, node_name=node_name, transform=transform)
    if scene.is_empty:
        raise ValueError("The USD file doesn't contain any visible mesh.")
    return scene


def _prototype_path(prim: Any) -> str:
    """The path of a prim, or of the prim of its prototype if it's in an instance: the same for all the instances."""
    return str(prim.GetPrimInPrototype().GetPath() if prim.IsInstanceProxy() else prim.GetPath())


def _is_rendered(imageable: Any, time: Any) -> bool:
    """Is a prim visible in a render: not invisible, and not a guide or a proxy (like a collision mesh)?"""
    from pxr import UsdGeom

    return imageable.ComputeVisibility(time) != UsdGeom.Tokens.invisible and imageable.ComputePurpose() in (
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
    )


def _convert_mesh(usd_mesh: Any, time: Any) -> list["trimesh.Trimesh"]:
    """Convert a USD mesh to triangle meshes, one per material of its faces (see UsdGeom.Subset)."""
    from pxr import UsdGeom, UsdShade

    points = np.asarray(usd_mesh.GetPointsAttr().Get(time) or [], dtype=np.float64).reshape(-1, 3)
    counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get(time) or [], dtype=np.int64)
    indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(time) or [], dtype=np.int64)
    if len(points) == 0 or len(counts) == 0 or counts.min() < 0 or counts.sum() != len(indices):
        return []
    if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
        return []

    # the polygons are triangulated as fans: the triangle k of a polygon has its corners 0, k + 1 and k + 2 (the
    # corners are the items of faceVertexIndices)
    triangle_counts = np.maximum(counts - 2, 0)
    polygon_of_triangles = np.repeat(np.arange(len(counts)), triangle_counts)
    first_triangles = np.cumsum(triangle_counts) - triangle_counts
    k = np.arange(triangle_counts.sum()) - first_triangles[polygon_of_triangles]
    starts = (np.cumsum(counts) - counts)[polygon_of_triangles]
    corners = np.stack([starts, starts + k + 1, starts + k + 2], axis=1)
    if usd_mesh.GetOrientationAttr().Get(time) == UsdGeom.Tokens.leftHanded:
        corners = corners[:, ::-1]

    # the authored normals keep the hard edges of the mesh: without them, the normals are smoothed over the faces
    # which share a point
    normals_primvar = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).GetPrimvar("normals")
    if normals_primvar and normals_primvar.HasValue():
        normals, interpolation = normals_primvar.ComputeFlattened(time), normals_primvar.GetInterpolation()
    else:
        normals, interpolation = usd_mesh.GetNormalsAttr().Get(time), usd_mesh.GetNormalsInterpolation()
    normals = np.asarray(normals if normals is not None else [], dtype=np.float64).reshape(-1, 3)
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(normals) == len(points):
        vertices, vertex_normals, triangles = points, normals, indices[corners]
    elif (interpolation == UsdGeom.Tokens.faceVarying and len(normals) == len(indices)) or (
        interpolation == UsdGeom.Tokens.uniform and len(normals) == len(counts)
    ):
        # normals per corner or per face: a vertex per point and normal, so that only the hard edges are split
        if interpolation == UsdGeom.Tokens.uniform:
            normals = normals[np.repeat(np.arange(len(counts)), counts)]
        vertex_keys, vertex_of_corners = np.unique(np.column_stack([indices, normals]), axis=0, return_inverse=True)
        vertices, vertex_normals = points[vertex_keys[:, 0].astype(np.int64)], vertex_keys[:, 1:]
        triangles = vertex_of_corners.reshape(-1)[corners]
    else:
        vertices, vertex_normals, triangles = points, None, indices[corners]

    binding = UsdShade.MaterialBindingAPI(usd_mesh.GetPrim())
    mesh_color = _material_color(binding.ComputeBoundMaterial()[0]) or _display_color(usd_mesh, time)
    # the faces of a subset have the material bound to the subset, the other faces have the one of the mesh
    colors, polygon_colors = [mesh_color], np.zeros(len(counts), dtype=np.int64)
    for subset in binding.GetMaterialBindSubsets():
        subset_color = _material_color(UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0])
        polygons = np.asarray(subset.GetIndicesAttr().Get(time) or [], dtype=np.int64)
        if subset_color:
            polygon_colors[polygons[(polygons >= 0) & (polygons < len(counts))]] = len(colors)
            colors.append(subset_color)
    triangle_colors = polygon_colors[polygon_of_triangles]
    return [
        _colored_mesh(vertices, vertex_normals, triangles[triangle_colors == color_index], color)
        for color_index, color in enumerate(colors)
        if (triangle_colors == color_index).any()
    ]


def _colored_mesh(
    vertices: np.ndarray, vertex_normals: Optional[np.ndarray], triangles: np.ndarray, color: tuple[float, ...]
) -> "trimesh.Trimesh":
    """Return the mesh of some triangles, with only their vertices, and a constant material."""
    import trimesh

    used_vertices, faces = np.unique(triangles, return_inverse=True)
    # process=False keeps the vertices as they are: trimesh would merge them, and lose the hard edges
    mesh = trimesh.Trimesh(
        vertices=vertices[used_vertices],
        faces=faces.reshape(-1, 3),
        vertex_normals=None if vertex_normals is None else vertex_normals[used_vertices],
        process=False,
    )
    *rgb, metallic, roughness, opacity = color
    material = trimesh.visual.material.PBRMaterial(  # type: ignore[no-untyped-call]
        baseColorFactor=[*rgb, opacity],
        metallicFactor=metallic,
        roughnessFactor=roughness,
        alphaMode="BLEND" if opacity < 1 else "OPAQUE",
        # the winding of the faces of many USD files is inconsistent
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)  # type: ignore[no-untyped-call]
    return mesh


def _material_color(material: Any) -> Optional[tuple[float, ...]]:
    """Return the constant (red, green, blue, metallic, roughness, opacity) of a UsdPreviewSurface material.

    None if the material is not a UsdPreviewSurface, or if its diffuse color comes from a texture.
    """
    from pxr import UsdShade

    if not material:
        return None
    shader = material.ComputeSurfaceSource()[0]
    if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
        return None

    def constant(name: str, default: Any) -> Any:
        shader_input = shader.GetInput(name)
        if not shader_input:
            return default
        # the value can be authored on the shader, or on an input of the material it's connected to: it's not a
        # constant if it's the output of another shader, like a texture
        attributes = shader_input.GetValueProducingAttributes()
        if not attributes or not UsdShade.Input.IsInput(attributes[0]):
            return None
        value = attributes[0].Get()
        return default if value is None else value

    diffuse_color = constant("diffuseColor", (0.18, 0.18, 0.18))
    if diffuse_color is None:
        return None
    metallic, roughness, opacity = constant("metallic", 0.0), constant("roughness", 0.5), constant("opacity", 1.0)
    return (
        *(float(c) for c in diffuse_color),
        float(metallic or 0.0),
        float(0.5 if roughness is None else roughness),
        float(1.0 if opacity is None else opacity),
    )


def _display_color(usd_mesh: Any, time: Any) -> tuple[float, ...]:
    """Return the color of a mesh without material: its displayColor, or a default gray."""
    colors = usd_mesh.GetDisplayColorPrimvar().Get(time)
    color = tuple(float(c) for c in colors[0]) if colors else USD_DEFAULT_COLOR
    return (*color, 0.0, 0.5, 1.0)
