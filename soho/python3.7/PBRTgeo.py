import os
import re
import array
import shlex
import subprocess
import collections

import hou
import soho

import PBRTapi as api
from PBRTnodes import BaseNode, MaterialNode, PBRTParam, ParamSet
from PBRTshading import wrangle_shading_network
from PBRTstate import scene_state, temporary_file, HVER_17_5, HVER_18


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


# TODO: Write a find_attrib_value(name, type, size)
#   this way we can scope the exact attribute we want
#   instead of getting a string when we want a float.
#   Update zmin_attrib = gdp.findPrimAttrib("zmin")
#   as an example

# NOTE: HOUDINI COMPATIBILITY
#   We can match Houdini's Sphere's with a 1,1,-1 Scale.
def sphere_wrangler(gdp, paramset=None, properties=None):
    """Outputs a "sphere" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    zmin_attrib = gdp.findPrimAttrib("zmin")
    zmax_attrib = gdp.findPrimAttrib("zmax")
    phimax_attrib = gdp.findPrimAttrib("phimax")

    match_uvs = True
    if (
        properties
        and "pbrt_matchhoudiniuv" in properties
        and not properties["pbrt_matchhoudiniuv"].Value[0]
    ):
        match_uvs = False

    with api.AttributeBlock():
        # Because we are inverting z-axis per sphere, we need to reverse orientation

        if match_uvs:
            api.ReverseOrientation()
        for prim in gdp.prims():
            shape_paramset = ParamSet(paramset)

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

            with api.AttributeBlock():
                xform = prim_transform(prim)
                api.ConcatTransform(xform)
                if match_uvs:
                    # Scale required to match Houdini's uvs
                    api.Scale(1, 1, -1)
                api.Shape("sphere", shape_paramset)
    return


# NOTE: HOUDINI COMPATIBILITY
#   The parameteric uvs do not match between the two. The u coordinate is
#   flipped. This is not resolvable within the export.
def disk_wrangler(gdp, paramset=None, properties=None):
    """Outputs "disk" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    # NOTE we could hou.math.clamp our radius and phimax here, but instead will let the
    # user pass them as is and let pbrt-v4 deal with it. The reasoning for this is that
    # this is slightly more advanced and we would expect the user to know what they are
    # doing.
    innerradius_attrib = gdp.findPrimAttrib("innerradius")
    phimax_attrib = gdp.findPrimAttrib("phimax")

    for prim in gdp.prims():
        shape_paramset = ParamSet(paramset)

        if innerradius_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "innerradius", prim.attribValue(innerradius_attrib))
            )
        if phimax_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "phimax", prim.attribValue(phimax_attrib))
            )

        with api.AttributeBlock():
            xform = prim_transform(prim)
            api.ConcatTransform(xform)
            api.Shape("disk", shape_paramset)
    return


def packeddisk_wrangler(gdp, paramset=None, properties=None):
    """Outputs "ply" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """
    for prim in gdp.prims():
        shape_paramset = ParamSet(paramset)
        filename = prim.intrinsicValue("filename")
        if not filename:
            continue
        if os.path.splitext(filename)[1].lower() != ".ply":
            continue
        shape_paramset.replace(PBRTParam("string", "filename", filename))
        with api.AttributeBlock():
            xform = prim_transform(prim)
            api.ConcatTransform(xform)
            api.Shape("plymesh", shape_paramset)
    return


def tube_wrangler(gdp, paramset=None, properties=None):
    """Handles "cylinder" Shapes for the input geometry

    Args:
        gdp (hou.Geometry): Input geo
        paramset (ParamSet): Any base params to add to the shape. (Optional)
        properties (dict): Dictionary of SohoParms (Optional)
    Returns: None
    """

    for prim in gdp.prims():

        shape_paramset = ParamSet(paramset)

        phimax_attrib = gdp.findPrimAttrib("phimax")
        if phimax_attrib is not None:
            shape_paramset.add(
                PBRTParam("float", "phimax", prim.attribValue(phimax_attrib))
            )

        with api.AttributeBlock():

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
                # NOTE: We are disabling this so that phimax will line up
                #       between the disks and the cylinder. This means Houdini's UV's
                #       will not match, but that is preferred over non-aligned
                #       disks and cylinders
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
def mesh_wrangler(gdp, paramset=None, properties=None):
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

    unique_points = True if to_promote else False

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
    P.frombytes(mesh_gdp.pointFloatAttribValuesAsString("P"))
    mesh_paramset.add(PBRTParam("point", "P", P))

    if N_attrib is not None:
        N = array.array("f")
        N.frombytes(mesh_gdp.pointFloatAttribValuesAsString("N"))
        mesh_paramset.add(PBRTParam("normal", "N", N))

    if S_attrib is not None:
        S = array.array("f")
        S.frombytes(mesh_gdp.pointFloatAttribValuesAsString("S"))
        mesh_paramset.add(PBRTParam("vector", "S", S))

    if faceIndices_attrib is not None:
        faceIndices = array.array("i")
        faceIndices.frombytes(mesh_gdp.primIntAttribValuesAsString("faceIndices"))
        mesh_paramset.add(PBRTParam("integer", "faceIndices", faceIndices))

    if uv_attrib is not None:
        uv = array.array("f")
        uv.frombytes(mesh_gdp.pointFloatAttribValuesAsString("uv"))
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
    P.frombytes(mesh_gdp.pointFloatAttribValuesAsString("P"))

    vertices = mesh_vtx_gen(mesh_gdp)
    indices = vtx_attrib_gen(vertices, None)

    mesh_paramset.add(PBRTParam("integer", "indices", indices))
    mesh_paramset.add(PBRTParam("point", "P", P))

    return mesh_paramset


# NOTE: HOUDINI COMPATIBILITY
#   see comment at patch_vtx_gen()
def patch_wrangler(gdp, paramset=None, properties=None):
    if properties is None:
        properties = {}

    blast_verb = hou.sopNodeTypeCategory().nodeVerb("blast")
    blast_verb.setParms({"group": '@intrinsic:connectivity!="quads"', "grouptype": 4})
    blast_verb.execute(gdp, [gdp])

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
    for medium, medium_prims in mediums_map.items():
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


def vdb_wrangler(gdp, paramset=None, properties=None):

    medium_paramset = ParamSet(paramset)

    # Perform a series of checks to see if we have a valid VDB
    if properties is None:
        properties = {}
        sop_path = None
    else:
        sop_path = properties["object:soppath"].Value[0]

    if not scene_state.allow_geofiles:
        return None

    if not scene_state.nanovdb_converter:
        return None

    if "pbrt_ignorevolumes" in properties and properties["pbrt_ignorevolumes"].Value[0]:
        api.Comment("Ignoring volumes because pbrt_ignorevolumes is enabled")
        return None

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

        # NOTE: We may want to reevaluate this with user testing and possibly use
        # tempdirs for the .vdb files instead?
        # See PBRTstate.get_geo_path_and_part for more details but the gist of this
        # we have 3 paths, the vdb path Houdini saves to. The nvdb which gain from
        # converting from the vdb path. Then ultimately the path of the nvdb file
        # in the pbrt scene file which might be a different relative path to the
        # one we exported.
        save_locations = scene_state.get_geo_path_and_part(sop_path, "nvdb")

        nvdb_path = save_locations.save_path

        pbrt_geo_dir = os.path.dirname(save_locations.pbrt_path)
        pbrt_geo_dir = "." if not pbrt_geo_dir else pbrt_geo_dir

        nvdb_basename = os.path.basename(nvdb_path)
        # Can't use os.path.join due to Houdini's / use on Windows
        pbrt_nvdb_path = "{}/{}".format(pbrt_geo_dir, nvdb_basename)

        if (
            "{vdb}" not in scene_state.nanovdb_converter
            or "{nanovdb}" not in scene_state.nanovdb_converter
        ):
            soho.error("'OpenVDB->NanoVDB Tool' needs {vdb} and {nanovdb} tokens")
            medium_gdp.clear()
            return None

        soho.makeFilePathDirsIfEnabled(nvdb_path)

        with temporary_file(suffix=".vdb") as vdb_path:
            convert_str = scene_state.nanovdb_converter.format(
                vdb=vdb_path, nanovdb=nvdb_path
            )

            # We could pass the full string, but I prefer to use a list
            convert_args = shlex.split(convert_str)

            medium_gdp.saveToFile(vdb_path)
            try:
                proc = subprocess.Popen(
                    convert_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except OSError:
                soho.error(
                    "Failed to run {}\n"
                    "Convert String: {}\n"
                    "You can disable VDB conversion by setting the"
                    "'OpenVDB->NanoVDB Tool' to be empty on the PBRT ROP".format(
                        convert_args[0], convert_str
                    )
                )
                medium_gdp.clear()
                return None

            stdout, stderr = proc.communicate()

            if proc.returncode:
                soho.error(
                    "Failed to run {}\n"
                    "Convert String: {}\n"
                    "Convert Error: {}\n"
                    "You can disable VDB conversion by setting the"
                    "'OpenVDB->NanoVDB Tool' to be empty on the PBRT ROP".format(
                        convert_args[0], convert_str, stderr
                    )
                )
                medium_gdp.clear()
                return None

        vdb_paramset = ParamSet()

        # NOTE Full Point instancing might be an issue when using VDBs.
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
    for medium, medium_prims in mediums_map.items():

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


def volume_wrangler(gdp, paramset=None, properties=None):

    # Houdini only supports one type of Volume primitive to be in a geometry network.
    # So if both a heightfield and a fog volume exists, the heightfield takes priority
    # and the fog volumes are ignored. We'll follow that logic here, but instead of
    # rendering the heightfield or volume we'll exit out since pbrt-v4 does not support
    # heightfields.

    if properties is None:
        properties = {}
        sop_path = None
    else:
        sop_path = properties["object:soppath"].Value[0]

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

    if medium_paramset.find_param("string", "preset") is None:
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


def smoke_prim_wrangler(grids, paramset=None, properties=None):
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
        sop_path = None
    else:
        sop_path = properties["object:soppath"].Value[0]

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
            voxeldata.frombytes(grid.density.allVoxelsAsString())
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
            r_voxeldata.frombytes(grid.r.allVoxelsAsString())
            g_voxeldata.frombytes(grid.g.allVoxelsAsString())
            b_voxeldata.frombytes(grid.b.allVoxelsAsString())
            voxeldata = array.array("f", r_voxeldata + g_voxeldata + b_voxeldata)
            voxeldata[0::3] = r_voxeldata
            voxeldata[1::3] = g_voxeldata
            voxeldata[2::3] = b_voxeldata
            smoke_paramset.add(PBRTParam("rgb", "density", voxeldata))

        medium_name = "%s[%s]%s" % (sop_path, prim_num_str, medium_suffix)

        if grid.lescale is not None:
            lescale_voxeldata = array.array("f")
            lescale_voxeldata.frombytes(grid.lescale.allVoxelsAsString())
            smoke_paramset.add(PBRTParam("float", "Lescale", lescale_voxeldata))

        resolution = ref_prim.resolution()

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
def curve_wrangler(gdp, paramset=None, properties=None):
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
        sop_path = None
    else:
        sop_path = properties["object:soppath"].Value[0]

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
        # NOTE: We could possibly convert the curves to a format that
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
                # NOTE: pbrt-v4 requires normals when rendering ribbon curves. If the
                # user didn't supply them we'll set them here
                soho.warning(
                    "{} has ribbon curves without normals, "
                    "defaulting to [0,0,1]".format(sop_path)
                )
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
        api.Shape("curve", curve_paramset)
    return


def tesselated_wrangler(gdp, paramset=None, properties=None):
    """Wrangler for any geo that needs to be tesselated"""
    prim_name = gdp.iterPrims()[0].intrinsicValue("typename")
    api.Comment(
        "%s prims is are not directly supported, they will be tesselated" % prim_name
    )
    mesh_wrangler(gdp, paramset, properties)
    return


def not_supported(gdp, paramset=None, properties=None):
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

def partition_by_intrinsic(input_gdp, intrinsic):
    """Partition the input geo based on a prim intrinsic

    Args:
        input_gdp (hou.Geometry): Incoming geometry, not modified
        intrinsic (str): Intrinsic name
    Returns:
        Dictionary of hou.Geometry with keys of the intrinsic value.
    """

    prim_values = set()
    for prim in input_gdp.iterPrims():
        prim_values.add(prim.intrinsicValue(intrinsic))

    split_gdps = {}

    blast_verb = hou.sopNodeTypeCategory().nodeVerbs()["blast"]

    needs_escape_pat = re.compile('([]["*?])')

    for prim_value in prim_values:
        escaped_value = needs_escape_pat.sub(r"\\\1", prim_value)
        blast_verb.setParms(
            {
                "negate": 1,
                "grouptype": 4,
                "group": '@intrinsic:{}="{}"'.format(intrinsic, escaped_value),
            }
        )

        gdp = hou.Geometry()
        blast_verb.execute(gdp, [input_gdp])
        split_gdps[prim_value] = gdp

    return split_gdps

def partition_by_attrib(input_gdp, attrib):
    """Partition the input geo based on a attribute

    Args:
        input_gdp (hou.Geometry): Incoming geometry, not modified
        attrib (str, hou.Attrib): Attribute to partition by
    Returns:
        Dictionary of hou.Geometry with keys of the attrib value.
    """

    attrib_name = attrib

    if isinstance(attrib, hou.Attrib):
        attrib_name = attrib.name()
    else:
        attrib = input_gdp.findPrimAttrib(attrib)

    if attrib.size() > 1:
        raise ValueError("Primitive attribute must be size 1")

    sort_verb = hou.sopNodeTypeCategory().nodeVerbs()["sort"]
    sort_verb.setParms( {"primsort":11, "primattrib":attrib_name} )
    sort_verb.execute(input_gdp, [input_gdp])

    if attrib.dataType() == hou.attribData.String:
        prim_values = input_gdp.primStringAttribValues(attrib_name)
    elif attrib.dataType() == hou.attribData.Int:
        prim_values = input_gdp.primIntAttribValues(attrib_name)
    elif attrib.dataType() == hou.attribData.Float:
        prim_values = input_gdp.primFloatAttribValues(attrib_name)
    else:
        raise ValueError("Invalid attribute type")

    split_gdps = {}

    def _put_in_cache(v, cache):
        if v in cache:
            return False
        cache.add(v)
        return True

    cache = set()
    run_lengths = [ (v,i) for i,v in enumerate(prim_values) if _put_in_cache(v,cache) ]
    run_lengths.append( (prim_values[-1], len(prim_values)) )

    blast_verb = hou.sopNodeTypeCategory().nodeVerbs()["blast"]

    for i,encoded_v in enumerate(run_lengths[:-1]):
        prim_value, start = encoded_v
        end = run_lengths[i+1][1]-1
        blast_verb.setParms(
            {
                "negate": 1,
                "grouptype": 4,
                "group": '{}-{}'.format(start, end),
            }
        )

        gdp = hou.Geometry()
        blast_verb.execute(gdp, [input_gdp])
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
        del gdp
    else:
        material_gdps = {default_material: gdp}

    # The gdp these point to may have been cleared
    del prim_override_h
    del prim_material_h

    for material, material_gdp in material_gdps.items():

        if material not in scene_state.shading_nodes:
            if material in scene_state.invalid_shading_nodes:
                api.Comment("Did not apply %s as it was not a PBRT material" % material)
            material_node = None
        else:
            api.AttributeBegin()
            api.NamedMaterial(material)
            material_node = MaterialNode(material)

        shape_gdps = partition_by_attrib(material_gdp, "typename", intrinsic=True)
        material_gdp.clear()
        del material_gdp

        for shape, shape_gdp in shape_gdps.items():

            # Aggregate overrides, instead of per prim
            if has_prim_overrides:
                override_attrib_h = shape_gdp.findPrimAttrib("material_override")
                override_gdps = partition_by_attrib(shape_gdp, override_attrib_h)
                shape_gdp.clear()
                del shape_gdp
                del override_attrib_h
            else:
                override_gdps = {default_override: shape_gdp}

            node_cache = {}
            param_cache = {}
            override_count = 0
            for override_str, override_gdp in override_gdps.items():

                base_paramset = ParamSet()
                base_paramset |= primitive_alpha_texs(properties)

                if override_str:
                    suffix = ":{}-{}".format(soppath, override_count)
                    api.AttributeBegin()
                    override_count += 1
                    overrides = eval(override_str, {}, {})

                    wrangle_shading_network(
                        material,
                        use_named=False,
                        exported_nodes=set(),
                        name_suffix=suffix,
                        overrides=overrides,
                        node_cache=node_cache,
                        param_cache=param_cache,
                    )

                # At this point the gdps are partitioned
                # * First by material
                # * Then by primitive type
                # * Lastly, for each different material override there is additional
                #   partitioning
                #
                # At this point we will NOT have varying types or materials within the
                # shape_wrangler.

                shape_wrangler = shape_wranglers.get(shape, not_supported)
                if shape_wrangler:
                    shape_wrangler(override_gdp, base_paramset, properties)
                override_gdp.clear()
                if override_str:
                    api.AttributeEnd()

        if material_node is not None:
            api.AttributeEnd()
    return
