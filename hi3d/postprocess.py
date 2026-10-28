"""Geometry-only mesh validation, filtering, simplification, and normalization."""

from __future__ import annotations

import logging
from numbers import Integral

import numpy as np

COMPONENT_FACE_RATIO = 0.005
"""Minimum fraction of total faces retained for a non-largest component."""


def validate_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """Validate a triangle mesh and return float64 vertices and int64 faces.

    Raises:
        ValueError: Geometry is empty, non-triangular, non-finite, out of range,
            or has zero extent.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("mesh must contain vertices shaped [V, 3]")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("mesh must contain triangular faces shaped [F, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must all be finite")
    faces = faces.astype(np.int64)
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("mesh faces contain out-of-range vertex indices")
    if float(np.max(np.ptp(vertices, axis=0))) == 0.0:
        raise ValueError("mesh must have non-zero extent")
    return vertices.copy(), faces.copy()


def face_components(faces: np.ndarray) -> np.ndarray:
    """Label faces by edge-connected component.

    Two faces are adjacent when they share an edge used by exactly those two faces,
    matching ``trimesh``'s ``face_adjacency``; non-manifold edges do not connect.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    face_count = len(faces)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges.sort(axis=1)
    edge_face = np.repeat(np.arange(face_count), 3)
    # One int64 key per sorted edge: a 1-D unique instead of a row-wise one.
    keys = edges[:, 0].astype(np.int64) * (int(edges.max()) + 1) + edges[:, 1]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    shared = counts[inverse] == 2
    order = np.argsort(inverse[shared], kind="stable")
    pairs = edge_face[shared][order].reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(face_count, face_count),
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def filter_components(vertices: np.ndarray, faces: np.ndarray, component_face_ratio: float):
    """Drop connected components holding less than ``component_face_ratio`` of all faces.

    The largest component is always kept; unused vertices are removed.
    """
    labels = face_components(faces)
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    keep = (labels == largest) | (sizes[labels] / len(faces) >= component_face_ratio)
    kept_faces = faces[keep]
    vertex_indices, inverse = np.unique(kept_faces.reshape(-1), return_inverse=True)
    return vertices[vertex_indices], inverse.reshape(-1, 3).astype(np.int64)


def simplify(vertices: np.ndarray, faces: np.ndarray, target_face_count: int):
    """Quadric simplification toward ``target_face_count`` faces with ``pyfqmr``.

    Returns the input unchanged, with a warning, when ``pyfqmr`` is not installed.
    """
    if len(faces) <= target_face_count:
        return vertices, faces
    try:
        import pyfqmr
    except ImportError:
        logging.warning(
            "Hi3D: pyfqmr is not installed, so the mesh is returned with %d faces "
            "instead of being simplified to %d. Install pyfqmr or use a "
            "decimation node.", len(faces), target_face_count,
        )
        return vertices, faces
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(
        target_count=int(target_face_count), aggressiveness=5, preserve_border=True, verbose=False
    )
    out_vertices, out_faces, _ = simplifier.getMesh()
    return np.asarray(out_vertices, dtype=np.float64), np.asarray(out_faces, dtype=np.int64)


def postprocess_mesh(
    vertices,
    faces,
    *,
    target_face_count: int,
    component_face_ratio: float = COMPONENT_FACE_RATIO,
):
    """Filter, simplify, center and unit-scale mesh geometry.

    Args:
        vertices: Vertex positions ``[V, 3]``.
        faces: Triangle indices ``[F, 3]``.
        target_face_count: Simplification target; ``0`` disables simplification.
        component_face_ratio: Minimum share of faces a non-largest component keeps.

    Returns:
        ``(vertices, faces, center, radius)``: the mesh centered on its vertex mean
        with unit maximum radius, and the mean and radius that were removed, so
        ``vertices * radius + center`` restores the pre-normalization frame.
    """
    if isinstance(target_face_count, bool) or not isinstance(target_face_count, Integral) or target_face_count < 0:
        raise ValueError(f"target_face_count must be a non-negative integer, got {target_face_count!r}")
    if not 0.0 <= component_face_ratio <= 1.0:
        raise ValueError(f"component_face_ratio must be in [0, 1], got {component_face_ratio!r}")

    vertices, faces = validate_mesh(vertices, faces)
    if component_face_ratio > 0.0:
        vertices, faces = filter_components(vertices, faces, float(component_face_ratio))
    if target_face_count > 0:
        vertices, faces = simplify(vertices, faces, int(target_face_count))
    vertices, faces = validate_mesh(vertices, faces)

    center = vertices.mean(axis=0, keepdims=True)
    vertices = vertices - center
    radius = float(np.linalg.norm(vertices, axis=1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("mesh cannot be normalized to a positive finite radius")
    return vertices / radius, faces, center.reshape(3), radius
