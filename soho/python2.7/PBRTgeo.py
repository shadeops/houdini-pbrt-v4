from __future__ import print_function, division, absolute_import

import os
import array
import shlex
import subprocess
import collections

import hou
import soho

import PBRTapi as api
from PBRTnodes import BaseNode, MaterialNode, PBRTParam, ParamSet
from PBRTstate import scene_state, HVER_17_5, HVER_18


def primitive_alpha_texs(properties):
    if not properties:
        return
    paramset = ParamSet()
    for prop in ("alpha",):
        if prop not in properties:
            continue
        tex = properties[prop].Value[0]
        if not tex:
            continue
        paramset.add(PBRTParam("texture", prop, tex))
    return paramset


# NOTE: HOUDINI COMPATIBILITY
#   To match Houdini's parametric uvs, we need to do
#   col,row ; col+1,row ; col,row+1 ; col+1,row+1
#   However this causes backfaces which breaks Mediums
#   so we'll need to api.ReverseOrientation
def patch_vtx_gen(gdp):
    for prim in gdp.iterPrims():
        for col in range(prim.numCols() - 1):
            for row in range(prim.numRows() - 1):
                yield prim.vertex(col, row)
                yield prim.vertex(col + 1, row)
                yield prim.vertex(col, row + 1)
                yield prim.vertex(col + 1, row + 1)


def mesh_vtx_gen(gdp):
    return (vtx for prim in gdp.iterPrims() for vtx in prim.vertices())


def vtx_attrib_gen(vertices, attrib):
    """Per prim, per vertex fetching vertex/point values

    Args:
        gdp (hou.Geometry): Input geometry
        attrib (hou.Attrib): Attribute to evaluate

    Yields:
        Values of attrib for each vertex
    """
    # NOTE: Having one loop with a conditional inside is a significant cost.
    #       We'll pull the conditional out of the loop so its computed once
    #       at the expense of some code dupilcation.
    if attrib is None:
        for vtx in vertices:
            yield vtx.point().number()
    elif attrib.type() == hou.attribType.Vertex:
        for vtx in vertices:
            yield vtx.attribValue(attrib)
    elif attrib.type() == hou.attribType.Point:
        for vtx in vertices:
            yield vtx.point().attribValue(attrib)


def linear_vtx_gen(gdp, vtx_per_face_hint=None):
    """Generate the linearvertex for input geometry

    A linear vertex is a unique value for every vertex in the mesh
    where as a vertex number is the vertex offset on a prim

    We need a linear vertex for generating indices when we have uniqe points
    http://www.sidefx.com/docs/houdini/vex/functions/vertexindex.html

    Args:
        gdp (hou.Geometry): Input geometry

    Yields:
        Linear vertex number for every vertex
    """
    # NOTE: The following can be skipped reduced down to a simple range since we
    #       know that the meshes will always have {vtx_per_face_hint} verts
    if vtx_per_face_hint is None:
        vertices = mesh_vtx_gen(gdp)
        return vtx_attrib_gen(vertices, None)

    return range(len(gdp.iterPrims()) * vtx_per_face_hint)


def prim_transform(prim):
    """Return a tuple representing the Matrix4 of the transform intrinsic"""
    # VDBs have a transform intrinsic that is a hou.Matrix4
    # rot_mat = hou.Matrix3(prim.intrinsicValue("transform"))
    # so we'll rely on the prim's transform() method
    rot_mat = prim.transform()
    vtx = prim.vertex(0)
    pt = vtx.point()
    pos = pt.position()
    xlate = hou.hmath.buildTranslate(pos)
    return (hou.Matrix4(rot_mat) * xlate).asTuple()


def prim_override(prim, override_node):
    # TODO pbrt-v4 does not support this
    paramset = ParamSet()
    return paramset

    if override_node is None:
        return paramset
    override = prim.attribValue("material_override")
    if not override:
        return paramset
    return override_node.override_paramset(override)


# TODO: Write a find_attrib_value(name, type, size)
#   this way we can scope the exact attribute we want
#   instead of getting a string when we want a float.
#   Update zmin_attrib = gdp.findPrimAttrib("zmin")
#   as an example

# NOTE: HOUDINI COMPATIBILITY
#   We can match Houdini's Sphere's with a 1,1,-1 Scale.
def sphere_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Outputs a "sphere" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    # TODO: Optimization, we can move the prim_attribs checks
    #       and api.Scale and api.ReverseOrientation() out of the for loop

    zmin_attrib = gdp.findPrimAttrib("zmin")
    zmax_attrib = gdp.findPrimAttrib("zmax")
    phimax_attrib = gdp.findPrimAttrib("phimax")

    for prim in gdp.prims():
        shape_paramset = ParamSet(paramset)
        shape_paramset |= prim_override(prim, override_node)

        if zmin_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "zmin", prim.attribValue(zmin_attrib))
            )
        if zmax_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "zmax", prim.attribValue(zmax_attrib))
            )
        if phimax_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "phimax", prim.attribValue(phimax_attrib))
            )

        with api.TransformBlock():
            xform = prim_transform(prim)
            api.ConcatTransform(xform)
            # Scale required to match Houdini's uvs
            api.Scale(1, 1, -1)
            # The inverted z-axis scale means we need to now reverse orientation
            api.ReverseOrientation()
            api.Shape("sphere", shape_paramset)
    return


# NOTE: HOUDINI COMPATIBILITY
#   The parameteric uvs do not match between the two. The u coordinate is
#   flipped. This is not resolvable within the export.
def disk_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Outputs "disk" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """
    # TODO: Since we don't set a radius due to using a transform
    # we might want to consider clamping innerradius to be < 1

    innerradius_attrib = gdp.findPrimAttrib("innerradius")
    phimax_attrib = gdp.findPrimAttrib("phimax")

    for prim in gdp.prims():
        shape_paramset = ParamSet(paramset)
        shape_paramset |= prim_override(prim, override_node)

        if innerradius_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "innerradius", prim.attribValue(innerradius_attrib))
            )
        if phimax_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "phimax", prim.attribValue(phimax_attrib))
            )

        with api.TransformBlock():
            xform = prim_transform(prim)
            api.ConcatTransform(xform)
            api.Shape("disk", shape_paramset)
    return


def packeddisk_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Outputs "ply" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """
    for prim in gdp.prims():
        shape_paramset = ParamSet(paramset)
        shape_paramset |= prim_override(prim, override_node)
        filename = prim.intrinsicValue("filename")
        if not filename:
            continue
        if os.path.splitext(filename)[1].lower() != ".ply":
            continue
        shape_paramset.replace(PBRTParam("string", "filename", filename))
        with api.TransformBlock():
            xform = prim_transform(prim)
            api.ConcatTransform(xform)
            api.Shape("plymesh", shape_paramset)
    return


def tube_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Handles "cylinder" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    for prim in gdp.prims():

        shape_paramset = ParamSet(paramset)
        shape_paramset |= prim_override(prim, override_node)

        phimax_attrib = gdp.findPrimAttrib("phimax")
        if phimax_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "phimax", prim.attribValue(phimax_attrib))
            )

        with api.TransformBlock():

            side_paramset = ParamSet(shape_paramset)

            xform = prim_transform(prim)
            taper = prim.intrinsicValue("tubetaper")

            if taper != 1:
                api.Comment(
                    "Skipping cylinder, prim #{}"
                    "taper values other than 1 not supported".format(prim.number())
                )
                continue

            closed = prim.intrinsicValue("closed")

            api.ConcatTransform(xform)
            api.Rotate(-90, 1, 0, 0)
            shape = "cylinder"
            side_paramset.add(PBRTParam("float", "zmin", -0.5))
            side_paramset.add(PBRTParam("float", "zmax", 0.5))

            with api.AttributeBlock():
                # Flip in Y so parameteric UV's match Houdini's
                # Note: We are disabling this so that phimax will line up
                #       between the disks and the cylinder
                # api.ReverseOrientation()
                # api.Scale(1, -1, 1)
                api.Shape(shape, side_paramset)

            if closed:
                disk_paramset = ParamSet(shape_paramset)
                disk_paramset.add(PBRTParam("float", "height", 0.5))
                api.Shape("disk", disk_paramset)
                disk_paramset.replace(PBRTParam("float", "height", -0.5))
                with api.AttributeBlock():
                    api.ReverseOrientation()
                    api.Shape("disk", disk_paramset)
    return


# NOTE: HOUDINI COMPATIBILITY
#   The parametric uvs for trianglemeshs do NOT match Houdini's. This is acceptable
#   since the common use case is to supply uvs. I believe it would be possible to
#   match the parametric uvs with pbrt however that means we'd lose the ability
#   to dump the various data arrays directly and slow things down.
def mesh_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Outputs meshes (trianglemesh or loopsubdiv) depending on properties

    If the pbrt_rendersubd property is set and true, a loopsubdiv shape will
    be generated, otherwise a trianglemesh

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    if properties is None:
        properties = {}

    mesh_paramset = ParamSet(paramset)

    # Triangle Meshes in PBRT uses "vertices" to denote positions.
    # These are similar to Houdini "points". Since the PBRT verts
    # are shared between primitives if hard edges or "vertex normals"
    # (Houdini-ese) are required then need to unique the points so
    # so each point can have its own normal.
    # To support this, if any of the triangle mesh params (N, uv, S)
    # are vertex attributes, then we'll uniquify the points.

    # We can only deal with triangles, where Houdini is a bit more
    # general, so we'll need to tesselate

    # If subdivs are turned on, instead of running the
    # trianglemesh wrangler, use the loop subdiv one instead

    shape = "trianglemesh"
    if "pbrt_rendersubd" in properties:
        if properties["pbrt_rendersubd"].Value[0]:
            shape = "loopsubdiv"

    gdp = scene_state.tesselate_geo(gdp)

    # Remove any open prims as they are not supported
    open_prims = [prim for prim in gdp.iterPrims() if not prim.intrinsicValue("closed")]
    gdp.deletePrims(open_prims)

    # Exit out if there are no prims
    if not any(gdp.iterPrims()):
        api.Comment("No primitives found")
        return None

    if shape == "loopsubdiv":
        wrangler_paramset = loopsubdiv_params(gdp)
        if "levels" in properties:
            wrangler_paramset.replace(properties["levels"].to_pbrt())
    else:
        computeN = True
        if "pbrt_computeN" in properties:
            computeN = properties["pbrt_computeN"].Value[0]
        wrangler_paramset = mesh_params(gdp, computeN)

    mesh_paramset.update(wrangler_paramset)

    api.Shape(shape, mesh_paramset)

    return None


def mesh_params(mesh_gdp, computeN=True, is_patchmesh=False):
    """Generates a ParamSet for a trianglemesh

    The following attributes are checked for -
     P (point), built-in attribute
     N (vertex/point), float[3]
     uv (vertex/point), float[3]
     S (vertex/point), float[3]
     faceIndices (prim), integer, used for ptex

    Args:
        mesh_gdp (hou.Geometry): Input geo
        computeN (bool): Whether to auto-compute normals if they don't exist
                         Defaults to True
    Returns: ParamSet of the attributes on the geometry
    """

    mesh_paramset = ParamSet()

    # Optional Attributes

    N_attrib = mesh_gdp.findVertexAttrib("N")
    if N_attrib is None:
        N_attrib = mesh_gdp.findPointAttrib("N")

    # If there are no vertex or point normals and we need to compute
    # them with a SopVerb
    if N_attrib is None and computeN:
        normal_verb = hou.sopNodeTypeCategory().nodeVerb("normal")
        # type 0 is point normals
        normal_verb.setParms({"type": 0})
        normal_verb.execute(mesh_gdp, [mesh_gdp])
        N_attrib = mesh_gdp.findPointAttrib("N")

    uv_attrib = mesh_gdp.findVertexAttrib("uv")
    if uv_attrib is None:
        uv_attrib = mesh_gdp.findPointAttrib("uv")

    if is_patchmesh:
        S_attrib = None
        faceIndices_attrib = None
    else:
        S_attrib = mesh_gdp.findVertexAttrib("S")
        if S_attrib is None:
            S_attrib = mesh_gdp.findPointAttrib("S")

        faceIndices_attrib = mesh_gdp.findPrimAttrib("faceIndices")

    # We need to unique the points if any of the handles
    # to vtx attributes exists.
    to_promote = []
    for attrib in (N_attrib, uv_attrib, S_attrib):
        if attrib is None:
            continue
        if attrib.type() == hou.attribType.Vertex:
            to_promote.append(attrib.name())

    unique_points = False
    if to_promote:
        unique_points = True

    if is_patchmesh:
        vertices = patch_vtx_gen(mesh_gdp)
    else:
        vertices = mesh_vtx_gen(mesh_gdp)

    if unique_points:
        if hou.applicationVersion() >= HVER_18:
            unique_verb = hou.sopNodeTypeCategory().nodeVerb("splitpoints")
        else:
            unique_verb = hou.sopNodeTypeCategory().nodeVerb("facet")
            unique_verb.setParms({"unique": True})
        unique_verb.execute(mesh_gdp, [mesh_gdp])

        promote_verb = hou.sopNodeTypeCategory().nodeVerb("attribpromote")
        # inclass 3 = vertex, method 8 = first match
        promote_str = " ".join(to_promote)
        promote_verb.setParms({"inclass": 3, "method": 8, "inname": promote_str})
        promote_verb.execute(mesh_gdp, [mesh_gdp])

        # If we sort the points by their vtx number we can just get a simple
        # range, the C++ Sort is much faster than looking up the actual point
        # numbers from the verts. The previous implementation of this was doing
        # the sort indirectly by iterator per vert per prim.
        if not is_patchmesh:
            sort_verb = hou.sopNodeTypeCategory().nodeVerb("sort")
            sort_verb.setParms({"ptsort": 1})
            sort_verb.execute(mesh_gdp, [mesh_gdp])

            indices = linear_vtx_gen(mesh_gdp, 3)
        else:
            indices = vtx_attrib_gen(vertices, None)

    else:
        indices = vtx_attrib_gen(vertices, None)

    mesh_paramset.add(PBRTParam("integer", "indices", indices))

    # NOTE: We are using arrays here for very fast access since we can
    #       fetch all the values at once compactly, while faster, this
    #       will take more RAM than a generator approach. If this becomes
    #       and issue we can change it.

    P = array.array("f")
    P.fromstring(mesh_gdp.pointFloatAttribValuesAsString("P"))
    mesh_paramset.add(PBRTParam("point", "P", P))

    if N_attrib is not None:
        N = array.array("f")
        N.fromstring(mesh_gdp.pointFloatAttribValuesAsString("N"))
        mesh_paramset.add(PBRTParam("normal", "N", N))

    if S_attrib is not None:
        S = array.array("f")
        S.fromstring(mesh_gdp.pointFloatAttribValuesAsString("S"))
        mesh_paramset.add(PBRTParam("vector", "S", S))

    if faceIndices_attrib is not None:
        faceIndices = array.array("i")
        faceIndices.fromstring(mesh_gdp.primIntAttribValuesAsString("faceIndices"))
        mesh_paramset.add(PBRTParam("integer", "faceIndices", faceIndices))

    if uv_attrib is not None:
        uv = array.array("f")
        uv.fromstring(mesh_gdp.pointFloatAttribValuesAsString("uv"))
        # Houdini's uvs are stored as 3 floats, but pbrt only needs two
        # We'll use some array slicing of continous memory to avoid
        # costly iteration
        # The follow is the equivalent of
        # uv_xy = (x for i, x in enumerate(uv) if i % 3 != 2)
        # but avoids having to do a mod for N times.
        uv_x = uv[::3]
        uv_y = uv[1::3]
        uv_xy = array.array("f", uv_x + uv_y)
        uv_xy[::2] = uv_x
        uv_xy[1::2] = uv_y
        mesh_paramset.add(PBRTParam("point2", "uv", uv_xy))

    return mesh_paramset


def loopsubdiv_params(mesh_gdp):
    """Generates a ParamSet for a loopsubdiv

    The following attributes are checked for -
    P (point), built-in attribute

    Args:
        mesh_gdp (hou.Geometry): Input geo
    Returns: ParamSet of the attributes on the geometry
    """

    mesh_paramset = ParamSet()

    P = array.array("f")
    P.fromstring(mesh_gdp.pointFloatAttribValuesAsString("P"))

    vertices = mesh_vtx_gen(mesh_gdp)
    indices = vtx_attrib_gen(vertices, None)

    mesh_paramset.add(PBRTParam("integer", "indices", indices))
    mesh_paramset.add(PBRTParam("point", "P", P))

    return mesh_paramset


# NOTE: HOUDINI COMPATIBILITY
#   see comment at patch_vtx_gen()
def patch_wrangler(gdp, paramset=None, properties=None, override_node=None):
    if properties is None:
        properties = {}

    # Remove any non quad prims as they are not supported
    non_quad_prims = [
        prim
        for prim in gdp.iterPrims()
        if prim.intrinsicValue("connectivity") != "quads"
    ]
    gdp.deletePrims(non_quad_prims)

    # Exit out if there are no prims
    if not any(gdp.iterPrims()):
        api.Comment("No primitives found")
        return None

    computeN = True
    if "pbrt_computeN" in properties:
        computeN = properties["pbrt_computeN"].Value[0]

    emission_attrib = gdp.findPrimAttrib("emissionfilename")
    if emission_attrib is None:
        if "pbrt_emissionfilename" in properties:
            emission_file = properties["pbrt_emissionfilename"].Value[0]
        else:
            emission_file = ""
        patch_gdps = {emission_file: gdp}
    else:
        patch_gdps = partition_by_attrib(gdp, emission_attrib)

    with api.AttributeBlock():
        api.ReverseOrientation()

        for emission_file, emission_gdp in patch_gdps.items():
            prim_paramset = ParamSet(paramset)
            if emission_file:
                prim_paramset.add(
                    PBRTParam("string", "emissionfilename", emission_file)
                )
            wrangler_paramset = mesh_params(gdp, computeN, is_patchmesh=True)
            prim_paramset.update(wrangler_paramset)

            api.Shape("bilinearmesh", prim_paramset)

    return None


class FloatGrid(object):
    def __init__(self, prim):
        self.density = prim
        self.lescale = None

    def does_res_match(self):
        if self.lescale is None:
            return True
        return self.density.resolution() == self.lescale.resolution()


class VDBGrid(object):
    def __init__(self, prim):
        self.density = prim
        self.temperature = None

    def does_res_match(self):
        return True


class RGBGrid(object):
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
        self.lescale = None

    def does_res_match(self):
        to_check = [self.r, self.g, self.b]
        if self.lescale is not None:
            to_check.append(self.lescale)
        return len(set([p.resolution() for p in to_check])) == 1


def build_vdb_grid_list(sop_path, gdp):
    prims = gdp.prims()

    name_attrib = gdp.findPrimAttrib("name")
    medium_grids_attrib = gdp.findPrimAttrib("medium_grids")

    name_map = collections.defaultdict(set)
    mediums_map = collections.defaultdict(set)
    for prim in prims:
        name_map[prim.attribValue(name_attrib)].add(prim)
        medium_name = ""
        if medium_grids_attrib is not None:
            medium_name = prim.attribValue(medium_grids_attrib)
        mediums_map[medium_name].add(prim)

    # The only senarios which are valid are-

    # Scenario 1
    # We do not have any secondary fields so we don't have to worry about linking
    # density fields to temperature
    name_counts = collections.Counter(prim.attribValue(name_attrib) for prim in prims)
    if not name_counts["temperature"]:
        return [VDBGrid(prim) for prim in prims]

    # Scenario 2
    # If a single temperature field and a density field exist but not a medium_grids
    # attribute we will infer that they are meant to be linked.
    if (
        name_counts["density"] == 1
        and name_counts["temperature"] == 1
        and medium_grids_attrib is None
    ):
        density_prim = gdp.globPrims("@name=density")[0]
        temperature_prim = gdp.globPrims("@name=temperature")[0]
        vdb = VDBGrid(density_prim)
        vdb.temperature = temperature_prim
        return [vdb]

    # Scenario 3
    # We have a medium_grids attribute and for each unqiue value we have only
    # one density, and 0 or 1 of our secondary field
    # But first we'll exit out if some of these conditions are not met.
    if (
        name_counts["density"] > 1
        and name_counts["temperature"] > 1
        and medium_grids_attrib is None
    ):
        soho.warning(
            "{}: has multiple density and temperature VDBs and no way to link"
            " them, please delete the temperature fields or use a "
            '"medium_grids" attribute'.format(sop_path)
        )
        return []

    grids = []
    for medium, medium_prims in mediums_map.iteritems():
        medium_counts = collections.Counter(
            prim.attribValue(name_attrib) for prim in medium_prims
        )
        # Issue warnings for poorly defined medium_grids.
        # The only cases that are valid are, 1 density to 0 or 1 temperature grids.
        # Or multiple density grids and 0 temperature grids
        if medium_counts["temperature"] > 1:
            soho.warning(
                "{}: has multiple temperature VDBs in a {}".format(sop_path, medium)
            )
            continue
        if medium_counts["density"] > 1 and medium_counts["temperature"]:
            soho.warning(
                "{}: the medium_grid {} has a mismatch of density and "
                "temperature VDBs".format(sop_path, medium)
            )
            continue
        if not medium_counts["density"]:
            soho.warning(
                "{}: the medium_grid {} has no density VDB".format(sop_path, medium)
            )
            continue

        if medium_counts["density"] == 1 and medium_counts["temperature"] in (0, 1):
            density_prim = name_map["density"] & medium_prims
            if len(density_prim) != 1:
                soho.warning("{}: Invalid density and medium_grid".format(sop_path))
                continue
            density_grid = VDBGrid(density_prim.pop())
            temperature_prim = name_map["temperature"] & medium_prims
            if temperature_prim:
                density_grid.temperature = temperature_prim.pop()
            grids.append(density_grid)
        else:
            soho.warning("{}: density grid failure".format(sop_path))

    return grids


def vdb_wrangler(gdp, paramset=None, properties=None, override_node=None):

    medium_paramset = ParamSet(paramset)

    # Perform a series of checks to see if we have a valid VDB
    if properties is None:
        properties = {}

    if not scene_state.allow_geofiles:
        return None

    if not scene_state.nanovdb_converter:
        return None

    if "pbrt_ignorevolumes" in properties and properties["pbrt_ignorevolumes"].Value[0]:
        api.Comment("Ignoring volumes because pbrt_ignorevolumes is enabled")
        return None

    sop_path = properties["object:soppath"].Value[0]

    name_attrib = gdp.findPrimAttrib("name")
    if name_attrib is None:
        soho.warning("Skipping {}, VDB prims do not have name attrib".format(sop_path))
        return None

    non_vdb_prims = gdp.globPrims("@name!=density,temperature")
    gdp.deletePrims(non_vdb_prims)

    if not gdp.prims():
        soho.warning(
            "Skipping {}, No VDBs prims named 'density' or 'temperature' found.".format(
                sop_path
            )
        )
        return None

    medium_grids = build_vdb_grid_list(sop_path, gdp)
    if not medium_grids:
        return None

    for medium_grid in medium_grids:

        # Cull prims we are not interested in
        grid_prim_numbers = set([medium_grid.density.number()])
        if medium_grid.temperature is not None:
            grid_prim_numbers.add(medium_grid.temperature.number())

        medium_gdp = hou.Geometry(gdp)
        cull_prims = [
            prim
            for prim in medium_gdp.iterPrims()
            if prim.number() not in grid_prim_numbers
        ]
        medium_gdp.deletePrims(cull_prims)

        bbox = medium_gdp.boundingBox()

        # TODO replace vdb_path with tempdir/?
        # vdb_path, part = scene_state.get_geo_path_and_part(sop_path, "vdb")
        save_locations = scene_state.get_geo_path_and_part(sop_path, "vdb")
        vdb_path = save_locations.save_path
        pathname = os.path.splitext(vdb_path)[0]
        nvdb_path = "{}.nvdb".format(pathname)
        nvdb_basename = os.path.basename(nvdb_path)
        pbrt_geo_dir = os.path.dirname(save_locations.pbrt_path)
        # Can't use os.path.join due to Houdini's / use on Windows
        pbrt_geo_dir = "." if not pbrt_geo_dir else pbrt_geo_dir
        pbrt_nvdb_path = "{}/{}".format(pbrt_geo_dir, nvdb_basename)

        if (
            "{vdb}" not in scene_state.nanovdb_converter
            or "{nanovdb}" not in scene_state.nanovdb_converter
        ):
            soho.error("'OpenVDB->NanoVDB Tool' needs {vdb} and {nanovdb} tokens")
            medium_gdp.clear()
            return None

        soho.makeFilePathDirsIfEnabled(vdb_path)
        convert_str = scene_state.nanovdb_converter.format(
            vdb=vdb_path, nanovdb=nvdb_path
        )
        convert_args = shlex.split(convert_str)
        medium_gdp.saveToFile(vdb_path)
        try:
            proc = subprocess.Popen(
                convert_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError:
            os.remove(vdb_path)
            soho.error(
                "Failed to run {}\nYou can disable VDB conversion by setting the"
                "'OpenVDB->NanoVDB Tool' to be empty on the PBRT ROP".format(
                    convert_args[0]
                )
            )
            medium_gdp.clear()
            return None

        stdout, stderr = proc.communicate()

        if proc.returncode:
            os.remove(vdb_path)
            soho.warning("Unable to convert VDB file: {}".format(vdb_path))
            medium_gdp.clear()
            return None

        vdb_paramset = ParamSet()

        # TODO Full Point instancing might be an issue when using VDBs.
        #       conversion might take forever, might need to cache based on
        #       parameters and reuse?
        vdb_paramset |= medium_paramset
        if "pbrt_interior" in properties:
            interior = BaseNode.from_node(properties["pbrt_interior"].Value[0])
            if interior is not None:
                if (
                    interior.directive_type != "nanovdb"
                    and interior.directive != "medium"
                ):
                    soho.warning(
                        "{} is not a valid medium for {}".format(
                            interior.node.path(), sop_path
                        )
                    )
                else:
                    vdb_paramset.update(interior.paramset)
            # These are special overrides that come from full point instancing.
            # It allows "per point" medium values to be "stamped" out to volume prims.
            interior_paramset = properties.get(".interior_overrides")
            if interior_paramset is not None:
                vdb_paramset.update(interior_paramset)

        exterior = None
        if "pbrt_exterior" in properties:
            exterior = properties["pbrt_exterior"].Value[0]
        exterior = "" if exterior is None else exterior

        extra_attribs = [
            ("float", "LeScale"),
            ("float", "temperaturecutoff"),
            ("float", "temperaturescale"),
        ]
        medium_prim_overrides = medium_prim_paramset(
            medium_grid.density, extra_attribs=extra_attribs
        )
        vdb_paramset.update(medium_prim_overrides)

        # By default we'll set a sigma_a and sigma_s to be more Houdini-like
        # however the object's pbrt_interior, or prim's pbrt_interior
        # or prim attribs will override these.

        if (
            PBRTParam("rgb", "sigma_a") not in vdb_paramset
            and PBRTParam("rgb", "sigma_s") not in vdb_paramset
        ) and PBRTParam("string", "preset") not in vdb_paramset:
            vdb_paramset.add(PBRTParam("spectrum", "sigma_a", [400.0, 0.0, 800.0, 0.0]))
            vdb_paramset.add(PBRTParam("spectrum", "sigma_s", [400.0, 1.0, 800.0, 1.0]))

        medium_suffix = ""
        instance_info = properties.get(".instance_info")
        if instance_info is not None:
            medium_suffix = ":%s[%i]" % (instance_info.source, instance_info.number)

        medium_name = "{}-{}{}".format(sop_path, save_locations.part, medium_suffix)

        vdb_paramset.replace(PBRTParam("string", "filename", pbrt_nvdb_path))
        with api.AttributeBlock():
            with api.TransformBlock():
                # xform = prim_transform(medium_grid.density)
                # api.ConcatTransform(xform)
                api.MakeNamedMedium(medium_name, "nanovdb", vdb_paramset)
                api.Material("none")
                api.MediumInterface(medium_name, exterior)
            vals = [x for pair in zip(bbox.minvec(), bbox.maxvec()) for x in pair]
            bounds_to_api_box(vals)

        medium_gdp.clear()

    return None


def build_uniform_grid_list(sop_path, gdp):
    prims = gdp.prims()

    name_attrib = gdp.findPrimAttrib("name")
    medium_grids_attrib = gdp.findPrimAttrib("medium_grids")

    # The only senarios which are valid are-
    # The first few senarios are convenience and don't enforce a medium_grids
    # attribute on the user

    # Scenario 1
    # No name attribute, we'll assume everything is density
    if name_attrib is None:
        return [FloatGrid(prim) for prim in prims]

    name_map = collections.defaultdict(set)
    mediums_map = collections.defaultdict(set)
    res_map = collections.defaultdict(set)

    density_renamer = {
        "density.x": "density.r",
        "density.y": "density.g",
        "density.z": "density.b",
    }

    name_counts = collections.defaultdict(int)
    for prim in prims:
        res = tuple(prim.resolution())
        res_map[res].add(prim)
        name = prim.attribValue(name_attrib)
        # Instead of dealing with two different variation of vectors
        # we'll rename to rgb
        name = density_renamer.get(name, name)
        name_counts[name] += 1
        name_map[name].add(prim)
        medium_name = ""
        if medium_grids_attrib is not None:
            medium_name = prim.attribValue(medium_grids_attrib)
        mediums_map[medium_name].add(prim)

    # Scenario 2
    # We just have density grids and no rgbs or Lescale
    if len(prims) == name_counts["density"]:
        return [FloatGrid(prim) for prim in prims]

    # Scenario 3
    # We just one density grid and one Lescale
    if (
        len(prims) == 2
        and name_counts["density"] == 1
        and name_counts["Lescale"]
        and medium_grids_attrib is None
    ):
        grid = FloatGrid(name_map["density"].pop())
        grid.lescale = name_map["Lescale"].pop()
        return [grid]

    grid_list = []
    den_rgb_strs = ("density.r", "density.g", "density.b")
    is_one_den_rgb = all(name_counts[c] == 1 for c in den_rgb_strs)
    is_no_den_rgb = all(name_counts[c] == 0 for c in den_rgb_strs)
    is_many_den_rgb = all(name_counts[c] > 1 for c in den_rgb_strs)

    # Scenario 4
    # We have no Lescale, but a mix of density and density.rgb
    # We can first remove all the density, then check the density.rgb
    if not name_counts["Lescale"] and (is_one_den_rgb or is_no_den_rgb):
        grid_list.extend([FloatGrid(x) for x in name_map["density"]])
        if is_one_den_rgb:
            grid_list.append(
                RGBGrid(
                    name_map["density.r"].pop(),
                    name_map["density.g"].pop(),
                    name_map["density.b"].pop(),
                )
            )
        return grid_list

    # Scenario 5
    # From this point on we won't be able to derive pairings without using the
    # medium_grids exit out if we don't fit bsaic requirements
    if (
        (name_counts["density"] > 1 and name_counts["Lescale"]) or is_many_den_rgb
    ) and medium_grids_attrib is None:
        soho.warning(
            "{}: has density or density.rgb/Lescale and no way to link"
            " them, please use a medium_grids attribute".format(sop_path)
        )
        return []

    grids = []
    for medium, medium_prims in mediums_map.iteritems():

        medium_counts = collections.defaultdict(int)
        for prim in medium_prims:
            name = prim.attribValue(name_attrib)
            name = density_renamer.get(name, name)
            medium_counts[name] += 1

        is_no_den_rgb = all(medium_counts[c] == 0 for c in den_rgb_strs)
        is_one_den_rgb = all(medium_counts[c] == 1 for c in den_rgb_strs)
        if (
            medium_counts["density"] == 1
            and medium_counts["Lescale"] <= 1
            and is_no_den_rgb
        ):
            density_prim = name_map["density"] & medium_prims
            if len(density_prim) != 1:
                soho.warning("{}: Invalid density and medium_grid".format(sop_path))
            density_grid = FloatGrid(density_prim.pop())
            lescale_prim = name_map["Lescale"] & medium_prims
            if lescale_prim:
                density_grid.lescale = lescale_prim.pop()
            grids.append(density_grid)
        elif (
            not medium_counts["density"]
            and medium_counts["Lescale"] <= 1
            and is_one_den_rgb
        ):
            r_prim = name_map["density.r"] & medium_prims
            g_prim = name_map["density.g"] & medium_prims
            b_prim = name_map["density.b"] & medium_prims
            if len(r_prim) != 1 or len(g_prim) != 1 or len(b_prim) != 1:
                soho.warning(
                    "{}: Invalid density.rgb and medium_grid {}".format(
                        sop_path, medium
                    )
                )
                continue
            rgb_grid = RGBGrid(r_prim.pop(), g_prim.pop(), b_prim.pop())
            lescale_prim = name_map["Lescale"] & medium_prims
            if lescale_prim:
                rgb_grid.lescale = lescale_prim.pop()
            grids.append(rgb_grid)
        else:
            soho.warning(
                "{}: Can not map density grids for {}".format(sop_path, medium)
            )

    return grids


def volume_wrangler(gdp, paramset=None, properties=None, override_node=None):

    # Houdini only supports one type of Volume primitive to be in a geometry network.
    # So if both a heightfield and a fog volume exists, the heightfield takes priority
    # and the fog volumes are ignored. We'll follow that logic here, but instead of
    # rendering the heightfield or volume we'll exit out since pbrt-v4 does not support
    # heightfields.

    if properties is None:
        properties = {}

    if "pbrt_ignorevolumes" in properties and properties["pbrt_ignorevolumes"].Value[0]:
        api.Comment("Ignoring volumes because pbrt_ignorevolumes is enabled")
        return None

    prims = gdp.prims()
    if any(prim.isHeightField() for prim in prims):
        api.Comment("Heightfields are not supported")
        return None

    # Filter out any SDFs as those are not supported either
    prims = [prim for prim in prims if not prim.isSDF()]

    # Houdini's Mantra's workflow for rendering volumes is a little difficult to match
    # to pbrt. Any fog volume primitives are rendered, with the density field by
    # default being used as the acceleration structure. (This can be overridden.)
    # This means you can have multiple density volumes and all are rendered. As those
    # are being rendered any other primitives with names that match parameters are
    # bound. Which means you can have multiple volume primitives with names like
    # temperature or Cd.[xyz]. Because of this it makes hard to match sets with volume
    # fields are associated with each other which we need to do when declaring mediums.
    # Potential Options include:
    #   * Volume Merge all same named fields into one.
    #   * Only render the first discovered density and cooresponding fields pbrt
    #       understands
    #   * Render all density volumes, the first found Cd gets mapped to the first
    #       found density
    #   * Match any volume prims that have the same x,y,z sample grid dimensions
    #       as density.
    # Further complicating things is pbrt can render either density or density.[rgb]
    #
    # The approach we will take is to require a medium_name attribute to group
    # prims together that form a medium. We'll derive some of the base mappings
    # automatically if the medium_name attribute does not exist.

    sop_path = properties["object:soppath"].Value[0]

    grids = build_uniform_grid_list(sop_path, gdp)
    smoke_prim_wrangler(grids, paramset, properties)

    return None


def bounds_to_api_box(b):
    """Output a trianglemesh Shape of box based on the input bounds"""

    paramset = ParamSet()
    paramset.add(
        PBRTParam(
            "point",
            "P",
            [
                b[1],
                b[2],
                b[5],
                b[0],
                b[2],
                b[5],
                b[1],
                b[3],
                b[5],
                b[0],
                b[3],
                b[5],
                b[0],
                b[2],
                b[4],
                b[1],
                b[2],
                b[4],
                b[0],
                b[3],
                b[4],
                b[1],
                b[3],
                b[4],
            ],
        )
    )
    paramset.add(
        PBRTParam(
            "integer",
            "indices",
            [
                0,
                3,
                1,
                0,
                2,
                3,
                4,
                7,
                5,
                4,
                6,
                7,
                6,
                2,
                7,
                6,
                3,
                2,
                5,
                1,
                4,
                5,
                0,
                1,
                5,
                2,
                0,
                5,
                7,
                2,
                1,
                6,
                4,
                1,
                3,
                6,
            ],
        )
    )
    api.Shape("trianglemesh", paramset)


# NOTE: In pbrt the medium interface and shading parameters
#       are strongly coupled unlike in Houdini/Mantra where
#       the volume shaders define the volume properties and
#       and the volume primitives only define grids.
#


def medium_prim_paramset(prim, paramset=None, extra_attribs=()):
    """Build a ParamSet of medium values based off of hou.Prim attribs"""
    medium_paramset = ParamSet(paramset)

    # NOTE:
    # Testing for prim attribs on each prim is a bit redundat but
    # in general its not an issue as you won't have huge numbers of
    # volumes. If this does become an issue, attribs can be stored in
    # a dict and searched from there. (This includes evaluating the
    # pbrt_interior node.

    # Initialize with the interior shader on the prim, if it exists.
    try:
        interior = prim.stringAttribValue("pbrt_interior")
        interior = BaseNode.from_node(interior)
    except hou.OperationFailed:
        interior = None

    # TODO We should check the directive type too? From the prim?
    #      or passed in by the wrangler?
    if interior and interior.directive == "medium":
        medium_paramset |= interior.paramset

    try:
        preset_value = prim.stringAttribValue("preset")
        if preset_value:
            medium_paramset.replace(PBRTParam("string", "preset", preset_value))
    except hou.OperationFailed:
        preset_value = None

    try:
        g_value = prim.floatAttribValue("g")
        medium_paramset.replace(PBRTParam("float", "g", g_value))
    except hou.OperationFailed:
        pass

    try:
        scale_value = prim.floatAttribValue("scale")
        medium_paramset.replace(PBRTParam("float", "scale", scale_value))
    except hou.OperationFailed:
        pass

    # TODO: What happens when we have a preset defined in pbrt_interior and
    #       a sigma_a or sigma_s supplied?
    if not preset_value:
        try:
            sigma_a_value = prim.floatListAttribValue("sigma_a")
            if len(sigma_a_value) == 3:
                medium_paramset.replace(PBRTParam("rgb", "sigma_a", sigma_a_value))
        except hou.OperationFailed:
            pass

        try:
            sigma_s_value = prim.floatListAttribValue("sigma_s")
            if len(sigma_s_value) == 3:
                medium_paramset.replace(PBRTParam("rgb", "sigma_s", sigma_s_value))
        except hou.OperationFailed:
            pass

    if not extra_attribs:
        return medium_paramset

    # This won't support every possible param types, just the
    # ones that are common (known)
    for attrib_type, name in extra_attribs:
        try:
            if attrib_type == "float":
                val = prim.floatAttribValue(name)
            elif attrib_type == "integer":
                val = prim.intAttribValue(name)
            elif attrib_type == "rgb":
                val = prim.floatListAttribValue(name)
                if len(val) != 3:
                    continue
            elif attrib_type == "string":
                val = prim.stringAttribValue(name)
            else:
                continue
        except hou.OperationFailed:
            continue
        medium_paramset.replace(PBRTParam(attrib_type, name, val))

    return medium_paramset


def smoke_prim_wrangler(grids, paramset=None, properties=None, override_node=None):
    """Outputs a "uniformgrid" Medium and bounding Shape for the input geometry

    The following attributes are checked for via medium_prim_paramset() -
    (See pbrt_medium node for what each parm does)
    pbrt_interior (prim), string
    preset (prim), string
    g (prim), float
    scale (prim), float
    sigma_a (prim), float[3]
    sigma_s (prim), float[3]

    Args:
        prims (list of hou.Prims): Input prims
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    if properties is None:
        properties = {}

    medium_paramset = ParamSet()
    if "pbrt_interior" in properties:
        interior = BaseNode.from_node(properties["pbrt_interior"].Value[0])
        if (
            interior is not None
            and interior.directive == "medium"
            and interior.directive_type == "uniformgrid"
        ):
            medium_paramset |= interior.paramset
        # These are special overrides that come from full point instancing.
        # It allows "per point" medium values to be "stamped" out to volume prims.
        interior_paramset = properties.get(".interior_overrides")
        if interior_paramset is not None:
            medium_paramset.update(interior_paramset)

    medium_suffix = ""
    instance_info = properties.get(".instance_info")
    if instance_info is not None:
        medium_suffix = ":%s[%i]" % (instance_info.source, instance_info.number)

    exterior = None
    if "pbrt_exterior" in properties:
        exterior = properties["pbrt_exterior"].Value[0]
    exterior = "" if exterior is None else exterior

    sop_path = properties["object:soppath"].Value[0]

    for grid in grids:
        smoke_paramset = ParamSet()

        if not grid.does_res_match():
            soho.warning(
                "{}: Skipping volumes that do not have matching resolutions".format(
                    sop_path
                )
            )
            continue

        if isinstance(grid, FloatGrid):
            prim_num_str = str(grid.density.number())
            ref_prim = grid.density
            voxeldata = array.array("f")
            voxeldata.fromstring(grid.density.allVoxelsAsString())
            smoke_paramset.add(PBRTParam("float", "density", voxeldata))
        else:
            prim_num_str = "{},{},{}".format(
                grid.r.number(), grid.g.number(), grid.b.number()
            )
            # We'll use the r|x channel as the reference
            ref_prim = grid.r

            r_voxeldata = array.array("f")
            g_voxeldata = array.array("f")
            b_voxeldata = array.array("f")
            r_voxeldata.fromstring(grid.r.allVoxelsAsString())
            g_voxeldata.fromstring(grid.g.allVoxelsAsString())
            b_voxeldata.fromstring(grid.b.allVoxelsAsString())
            voxeldata = array.array("f", r_voxeldata + g_voxeldata + b_voxeldata)
            voxeldata[0::3] = r_voxeldata
            voxeldata[1::3] = g_voxeldata
            voxeldata[2::3] = b_voxeldata
            smoke_paramset.add(PBRTParam("rgb", "density", voxeldata))

        medium_name = "%s[%s]%s" % (sop_path, prim_num_str, medium_suffix)

        if grid.lescale is not None:
            lescale_voxeldata = array.array("f")
            lescale_voxeldata.fromstring(grid.lescale.allVoxelsAsString())
            smoke_paramset.add(PBRTParam("float", "Lescale", lescale_voxeldata))

        resolution = ref_prim.resolution()
        # TODO: Benchmark this vs other methods like fetching volumeSlices

        smoke_paramset.add(PBRTParam("integer", "nx", resolution[0]))
        smoke_paramset.add(PBRTParam("integer", "ny", resolution[1]))
        smoke_paramset.add(PBRTParam("integer", "nz", resolution[2]))
        smoke_paramset.add(PBRTParam("point", "p0", [-1, -1, -1]))
        smoke_paramset.add(PBRTParam("point", "p1", [1, 1, 1]))

        extra_attribs = [("rgb", "Le")]
        medium_prim_overrides = medium_prim_paramset(
            ref_prim, medium_paramset, extra_attribs=extra_attribs
        )
        smoke_paramset.update(medium_prim_overrides)
        smoke_paramset |= paramset

        # By default we'll set a sigma_a and sigma_s to be more Houdini-like
        # however the object's pbrt_interior, or prim's pbrt_interior
        # or prim attribs will override these.
        if (
            PBRTParam("rgb", "sigma_a") not in smoke_paramset
            and PBRTParam("rgb", "sigma_s") not in smoke_paramset
        ) and PBRTParam("string", "preset") not in smoke_paramset:
            smoke_paramset.add(
                PBRTParam("spectrum", "sigma_a", [400.0, 0.0, 800.0, 0.0])
            )
            smoke_paramset.add(
                PBRTParam("spectrum", "sigma_s", [400.0, 1.0, 800.0, 1.0])
            )

        with api.AttributeBlock():
            xform = prim_transform(ref_prim)
            api.ConcatTransform(xform)
            api.MakeNamedMedium(medium_name, "uniformgrid", smoke_paramset)
            api.Material("none")
            api.MediumInterface(medium_name, exterior)
            # Pad this slightly?
            bounds_to_api_box([-1, 1, -1, 1, -1, 1])
    return


def _convert_nurbs_to_bezier(gdp):
    """Convert any NURBS Curves to Beziers

    Due to how knots are interrupted between Houdini and PBRT we won't be able to
    map NURBS to B-Splines. To work around this we just convert to Bezier degree 4
    curves, which is what PBRT is doing internally as well. "yolo"

    Args:
        gdp (hou.Geometry): Input geo
    Returns: None (Replaces input gdp)
    """

    # The Convert SOP is only available as a Verb in H17.5 and greater
    if hou.applicationVersion() < HVER_17_5:
        return

    convert_verb = hou.sopNodeTypeCategory().nodeVerb("convert")
    # fromtype: "nurbCurve", totype: "bezCurve"
    convert_verb.setParms({"fromtype": 9, "totype": 2})
    convert_verb.execute(gdp, [gdp])
    return


# NOTE: HOUDINI COMPATIBILITY
#   The parametric uvs on curves do not match Houdini, v is flipped.
def curve_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Outputs a "curve" Shape for input geometry

    The following attributes are checked for -

    P (point), built-in attribute
    width (vertex/point/prim), float
    N (vertex/point), float[3]
    curvetype (prim), string (overrides the property pbrt_curvetype)

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    if properties is None:
        properties = {}

    shape_paramset = ParamSet(paramset)

    curve_type = None
    if "pbrt_curvetype" in properties:
        curve_type = properties["pbrt_curvetype"].Value[0]
        shape_paramset.add(PBRTParam("string", "type", curve_type))
    if "splitdepth" in properties:
        shape_paramset.add(properties["splitdepth"].to_pbrt())

    _convert_nurbs_to_bezier(gdp)

    has_vtx_width = False if gdp.findVertexAttrib("width") is None else True
    has_pt_width = False if gdp.findPointAttrib("width") is None else True
    has_prim_width = False if gdp.findPrimAttrib("width") is None else True
    has_prim_width01 = False
    if (
        gdp.findPrimAttrib("width0") is not None
        and gdp.findPrimAttrib("width1") is not None
    ):
        has_prim_width01 = True

    has_curvetype = False if gdp.findPrimAttrib("curvetype") is None else True

    has_vtx_N = False if gdp.findVertexAttrib("N") is None else True
    has_pt_N = False if gdp.findPointAttrib("N") is None else True

    for prim in gdp.prims():

        curve_paramset = ParamSet()
        prim_curve_type = curve_type

        # Closed curve surfaces are not supported
        if prim.intrinsicValue("closed"):
            continue

        order = prim.intrinsicValue("order")
        degree = order - 1
        # PBRT only supports degree 2 or 3 curves
        # TODO: We could possibly convert the curves to a format that
        #       pbrt supports but for now we'll expect the user to have
        #       a curve basis which is supported
        # https://www.codeproject.com/Articles/996281/NURBS-crve-made-easy
        if degree not in (2, 3):
            continue
        curve_paramset.add(PBRTParam("integer", "degree", degree))

        if prim.intrinsicValue("typename") == "BezierCurve":
            basis = "bezier"
        else:
            # In Houdini 17.5 and greater we convert everything to bezier,
            # for Houdini 17 this isn't possible so we instead we skip them
            # basis = "bspline"
            continue
        curve_paramset.add(PBRTParam("string", "basis", [basis]))

        P = [pt.attribValue("P") for pt in prim.points()]
        curve_paramset.add(PBRTParam("point", "P", P))

        if has_curvetype:
            prim_val = prim.attribValue("curvetype")
            prim_curve_type = prim_val if prim_val else curve_type

        if prim_curve_type is not None:
            curve_paramset.add(PBRTParam("string", "type", [prim_curve_type]))

        if prim_curve_type == "ribbon":

            if has_vtx_N or has_pt_N:
                N = (prim.attribValueAt("N", u) for u in prim.intrinsicValue("knots"))
            else:
                # If ribbon, normals must exist
                # TODO: Let pbrt error? Or put default values?
                N = [(0, 0, 1)] * len(prim.intrinsicValue("knots"))

            if N is not None:
                curve_paramset.add(PBRTParam("normal", "N", N))

        if has_vtx_width:
            curve_paramset.add(
                PBRTParam("float", "width0", prim.vertex(0).attribValue("width"))
            )
            curve_paramset.add(
                PBRTParam("float", "width1", prim.vertex(-1).attribValue("width"))
            )
        elif has_pt_width:
            curve_paramset.add(
                PBRTParam(
                    "float", "width0", prim.vertex(0).point().attribValue("width")
                )
            )
            curve_paramset.add(
                PBRTParam(
                    "float", "width1", prim.vertex(-1).point().attribValue("width")
                )
            )
        elif has_prim_width01:
            curve_paramset.add(PBRTParam("float", "width0", prim.attribValue("width0")))
            curve_paramset.add(PBRTParam("float", "width1", prim.attribValue("width1")))
        elif has_prim_width:
            curve_paramset.add(PBRTParam("float", "width", prim.attribValue("width")))
        else:
            # Houdini's default matches a width of 0.05
            curve_paramset.add(PBRTParam("float", "width", 0.05))

        curve_paramset |= shape_paramset
        curve_paramset |= prim_override(prim, override_node)
        api.Shape("curve", curve_paramset)
    return


def tesselated_wrangler(gdp, paramset=None, properties=None, override_node=None):
    """Wrangler for any geo that needs to be tesselated"""
    prim_name = gdp.iterPrims()[0].intrinsicValue("typename")
    api.Comment(
        "%s prims is are not directly supported, they will be tesselated" % prim_name
    )
    mesh_wrangler(gdp, paramset, properties)
    return


def not_supported(gdp, paramset=None, properties=None, override_node=None):
    """Wrangler for unsupported prim types"""
    num_prims = len(gdp.iterPrims())
    prim_name = gdp.iterPrims()[0].intrinsicValue("typename")
    api.Comment("Ignoring %i prims, %s is not supported" % (num_prims, prim_name))
    return


shape_wranglers = {
    "Sphere": sphere_wrangler,
    "Circle": disk_wrangler,
    "Tube": tube_wrangler,
    "Poly": mesh_wrangler,
    "Mesh": patch_wrangler,
    "PolySoup": mesh_wrangler,
    "NURBMesh": tesselated_wrangler,
    "BezierCurve": curve_wrangler,
    "NURBCurve": curve_wrangler,
    "Volume": volume_wrangler,
    "VDB": vdb_wrangler,
    "PackedDisk": packeddisk_wrangler,
    "TriFan": tesselated_wrangler,
    "TriStrip": tesselated_wrangler,
    "TriBezier": tesselated_wrangler,
    "BezierMesh": tesselated_wrangler,
    "PasteSurf": tesselated_wrangler,
    "MetaBall": tesselated_wrangler,
    "MetaSQuad": tesselated_wrangler,
    "Tetrahedron": tesselated_wrangler,
}


# These are the types that the primtives form an aggregate.
# For example you can have a single polygon or combine multiple into
# a poly mesh. We'll want to combine the same overrides into a single
# mesh to save on creating a mesh per poly face.
def requires_override_partition(shape_type):
    return shape_wranglers[shape_type] in set([mesh_wrangler, tesselated_wrangler])


def partition_by_attrib(input_gdp, attrib, intrinsic=False):
    """Partition the input geo based on a attribute

    Args:
        input_gdp (hou.Geometry): Incoming geometry, not modified
        attrib (str, hou.Attrib): Attribute to partition by
        intrinsic (bool): Whether to an attribute or intrinsic attrib
                          (Optional, defaults to False)
    Returns:
        Dictionary of hou.Geometry with keys of the attrib value.
    """
    # Not sure about a set operation on prims
    prim_values = collections.defaultdict(set)
    prims = input_gdp.prims()
    if intrinsic:
        for prim in prims:
            prim_values[prim.intrinsicValue(attrib)].add(prim.number())
    else:
        for prim in prims:
            prim_values[prim.attribValue(attrib)].add(prim.number())

    split_gdps = {}
    all_prims = set(range(len(prims)))
    for prim_value in prim_values:
        gdp = hou.Geometry()
        gdp.merge(input_gdp)
        keep_prims = prim_values[prim_value]
        remove_prims = all_prims - keep_prims
        cull_list = [gdp.iterPrims()[p] for p in remove_prims]
        gdp.deletePrims(cull_list)
        split_gdps[prim_value] = gdp
    return split_gdps


def output_geo(soppath, now, properties=None):
    """Output the geometry by calling the appropriate wrangler

    Geometry is partitioned into subparts based on the shop_materialpath
    and material_override prim attributes.

    Args:
        soppath (str): oppath to SOP
        properties (dict, None): Dictionary of SohoParms
                                 (Optional, defaults to None)
    Returns:
        None
    """

    # split by material
    # split by geo type
    # if mesh type, split by material override
    # else deal with overrides per prim
    #
    # NOTE: We won't be splitting based on medium interior/exterior
    #       those will be left as a object level assignment only.
    #       Note, that in the case of Houdini Volumes they will look
    #       for the appropriate medium parameters as prim vars

    if properties is None:
        properties = {}

    ignore_materials = False
    if "pbrt_ignorematerials" in properties:
        ignore_materials = properties["pbrt_ignorematerials"].Value[0]

    # Houdini / Mantra allows for shop_materialpaths on both prims and details
    # at the same time. However prims full stomp over detail. If you have a prim
    # with an empty material assignment, it will NOT fall back to the detail
    # assignment. (It will fall back to the object since that is further up the
    # stack). This means if the shop_materialpath exists on the prim, the
    # detail is ignored entirely.

    # PBRT allows setting Material parameters on the Shapes in order to
    #       override a material's settings.  (Shapes get checked first)
    #       This paramset will be for holding those overrides and passing
    #       them down to the actual shape api calls.

    # We need the soppath to come along and since we are creating new
    # hou.Geometry() we'll lose the original sop connection so we need
    # to stash it here.

    node = hou.node(soppath)
    if node is None or node.type().category() != hou.sopNodeTypeCategory():
        return

    input_gdp = node.geometry()
    if input_gdp is None:
        return
    gdp = hou.Geometry()
    gdp.merge(input_gdp.freeze())

    default_material = ""
    default_override = ""
    if not ignore_materials:
        try:
            default_material = gdp.stringAttribValue("shop_materialpath")
        except hou.OperationFailed:
            pass
        if default_material not in scene_state.shading_nodes:
            default_material = ""

        try:
            default_override = gdp.stringAttribValue("material_override")
        except hou.OperationFailed:
            default_override = ""

    # These handles are only valid until until we clear the geo
    prim_material_h = gdp.findPrimAttrib("shop_materialpath")
    prim_override_h = gdp.findPrimAttrib("material_override")

    has_prim_overrides = bool(
        not ignore_materials
        and prim_override_h is not None
        and prim_material_h is not None
    )

    if prim_material_h is not None and not ignore_materials:
        material_gdps = partition_by_attrib(gdp, prim_material_h)
        gdp.clear()
    else:
        material_gdps = {default_material: gdp}

    # The gdp these point to may have been cleared
    del prim_override_h
    del prim_material_h

    for material, material_gdp in material_gdps.iteritems():

        if material not in scene_state.shading_nodes:
            if material in scene_state.invalid_shading_nodes:
                api.Comment("Did not apply %s as it was not a PBRT material" % material)
            material = ""
            material_node = None
        else:
            api.AttributeBegin()
            api.NamedMaterial(material)
            material_node = MaterialNode(material)

        shape_gdps = partition_by_attrib(material_gdp, "typename", intrinsic=True)
        material_gdp.clear()

        for shape, shape_gdp in shape_gdps.iteritems():

            # Aggregate overrides, instead of per prim
            if has_prim_overrides and requires_override_partition(shape):
                override_attrib_h = shape_gdp.findPrimAttrib("material_override")
                override_gdps = partition_by_attrib(shape_gdp, override_attrib_h)
                shape_gdp.clear()
                del override_attrib_h

                # We don't want the wranglers to handle the overrides since we are doing
                # it here. So we'll set this to false, which will mean the override_node
                # is None and not trigger per prim overrides
                has_prim_overrides = False
            else:
                override_gdps = {default_override: shape_gdp}

            for override, override_gdp in override_gdps.iteritems():

                base_paramset = ParamSet()
                # TODO pbrt-v4 material overrides are no longer suppored
                #      redo the aggreate vs prim sections with this in mind
                #      Adding the False to the conditional below until this
                #      is done.
                if override and material_node is not None and False:
                    # material parm overrides are only valid for MaterialNodes
                    base_paramset |= material_node.override_paramset(override)

                base_paramset |= primitive_alpha_texs(properties)

                if has_prim_overrides:
                    override_node = material_node
                else:
                    override_node = None

                # At this point the gdps are partitioned first by material
                # then by type. And then if its in requires_override_partition
                # it has been further partitioned.
                # The implies that we will NOT have varying types or materials
                # past this point. The wranglers will need to know the following-
                #   * is there a prim override?
                #   * is the material valid?
                #   * the material_node itself to apply overrides if they exist
                #
                #   The only case where we *need* to pass down the override info is if
                #   * the material_node is valid
                #   * material_overrides exists
                #
                #   We don't want to reconstruct a new material_node for every prim
                #   as it will be constant.
                #
                #   Option 1: <selected>
                #   We can pass a material_node only if we need to apply overrides
                #   but that gives variable dual meaning.
                #   Option 2:
                #   Alternatively we can pass the material_node and also a
                #   prim_overrides flag either in the properties or as its own function
                #   arg.

                shape_wrangler = shape_wranglers.get(shape, not_supported)
                if shape_wrangler:
                    shape_wrangler(
                        override_gdp, base_paramset, properties, override_node
                    )
                override_gdp.clear()

        if material:
            api.AttributeEnd()
    return
