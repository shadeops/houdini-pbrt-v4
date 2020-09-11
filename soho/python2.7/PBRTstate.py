from __future__ import print_function, division, absolute_import

import os
import collections

import hou
import soho

# Baseline Support is Houdini 17.0

# Houdini 17.5:
#   Supports Convert SOP as a Verb (this allows for the tesselator to be verb chain)
HVER_17_5 = (17, 5, 0)

# Houdini 18.0:
#   Changes from a Fuse SOP to a Split Points SOP
#   Support for getting full vertex lists directly from the gdp
HVER_18 = (18, 0, 0)


class PBRTState(object):
    """Holds the global state of the render session.

    This class can also act as a context, which will create helper networks
    and destory them on exit.
    """

    # TODO replace the cachedUserData workflow with a hou.Geometry parm
    _tesslate_py = """
node = hou.pwd()
geo = node.geometry()
geo.clear()
gdp = hou.node('..').cachedUserData('gdp')
if gdp is not None:
    geo.merge(gdp)
"""

    def __init__(self):
        self.shading_nodes = set()
        self.invalid_shading_nodes = set()
        self.medium_nodes = set()
        self.instanced_geo = set()
        self.geometry_parts = collections.defaultdict(int)
        # We do not interior/exterior these directly but are handy as
        # a quick way of seeing if they are set at the camera/rop
        # level
        self.interior = None
        self.exterior = None
        self.tesselator = None
        self.have_nanovdb_convert = True
        self.allow_geofiles = None
        self.geo_location = None
        self.nanovdb_converter = None
        self.geofile_threshold = None
        self.rop = None
        self.output_mode = None
        self.disk_file = None
        self.hip = None
        self.hipfile = None
        self.fps = None
        self.ver = None
        self.now = None

        self.inv_fps = None
        return

    def init_state(self):
        """Queries Soho to initialize the attributes of the class"""
        state_parms = {
            "rop": soho.SohoParm("object:name", "string", key="rop"),
            "soho_outputmode": soho.SohoParm(
                "soho_outputmode", "integer", skipdefault=False, key="output_mode"
            ),
            "soho_diskfile": soho.SohoParm(
                "soho_diskfile", "string", skipdefault=False, key="disk_file"
            ),
            "hip": soho.SohoParm("$HIP", "string", key="hip"),
            "hipname": soho.SohoParm("$HIPNAME", "string", key="hipname"),
            "hipfile": soho.SohoParm("$HIPFILE", "string", key="hipfile"),
            "ver": soho.SohoParm(
                "state:houdiniversion", "string", ["9.0"], False, key="ver"
            ),
            "now": soho.SohoParm("state:time", "real", [0], False, key="now"),
            "fps": soho.SohoParm("state:fps", "real", [24], False, key="fps"),
            "pbrt_allow_geofiles": soho.SohoParm(
                "pbrt_allow_geofiles", "bool", [1], False, key="allow_geofiles"
            ),
            "pbrt_geo_location": soho.SohoParm(
                "pbrt_geo_location",
                "string",
                ["$HIP/geometry"],
                False,
                key="geo_location",
            ),
            "pbrt_nanovdb_converter": soho.SohoParm(
                "pbrt_nanovdb_converter",
                "string",
                ["nanovdb_convert"],
                False,
                key="nanovdb_converter",
            ),
            "pbrt_geofile_threshold": soho.SohoParm(
                "pbrt_geofile_threshold", "integer", [10000], key="geofile_threshold"
            ),
        }
        rop = soho.getOutputDriver()
        parms = soho.evaluate(state_parms, None, rop)
        for parm in parms:
            setattr(self, parm, parms[parm].Value[0])
        if not self.fps:
            self.fps = 24.0
        self.inv_fps = 1.0 / self.fps
        return

    def __enter__(self):
        self.reset()
        self.init_state()
        self.tesselator = self.create_tesselator()
        return

    def __exit__(self, *args):
        self.reset()
        return

    @property
    def output_location(self):
        """Provide a relative geo location path if its a subdir of the disk file"""
        if self.output_mode != 1:
            return self.geo_location
        disk_file_path = os.path.dirname(self.disk_file)
        rel_geo_path = os.path.relpath(self.geo_location, disk_file_path)
        if rel_geo_path.startswith(".."):
            return self.geo_location
        return rel_geo_path

    def get_geo_path(self, sop_path, ext):
        part_num = self.geometry_parts[sop_path]
        self.geometry_parts[sop_path] += 1

        # TODO: We could be smart and not output frames that are not time dependent.
        #       we can check this via a sohog gdp.globalValue('geo:timedependent')[0]

        frame = hou.timeToFrame(self.now)
        filename = sop_path
        if filename.startswith("/obj/"):
            filename = filename[5:]
        filename = filename.replace("/", "-")
        geo_path = "{location}/{filename}-{part}.{frame:g}.{ext}".format(
            location=self.output_location,
            filename=filename,
            part=part_num,
            frame=frame,
            ext=ext,
        )
        return geo_path

    def reset(self):
        """Resets the class attributes back to their default state"""
        self.rop = None
        self.output_mode = None
        self.disk_file = None
        self.hip = None
        self.hipfile = None
        self.fps = None
        self.ver = None
        self.now = None
        self.inv_fps = None
        self.allow_geofiles = None
        self.geo_location = None
        self.nanovdb_converter = None
        self.geofile_threshold = None
        self.shading_nodes.clear()
        self.invalid_shading_nodes.clear()
        self.medium_nodes.clear()
        self.instanced_geo.clear()
        self.geometry_parts.clear()
        self.interior = None
        self.exterior = None
        self.have_nanovdb_convert = True
        self.remove_tesselator()
        return

    def tesselate_geo(self, geo):
        if hou.applicationVersion() >= HVER_17_5:
            return self.tesselate_geo_with_verbs(geo)
        return self.tesselate_geo_with_network(geo)

    def tesselate_geo_with_verbs(self, gdp):

        # Delete open primitives as PBRT does not support them
        convert_verb = hou.sopNodeTypeCategory().nodeVerb("convert")
        convert_verb.setParms({"lodu": 1, "lodv": 1})
        convert_verb.execute(gdp, [gdp])

        divide_verb = hou.sopNodeTypeCategory().nodeVerb("divide")
        divide_verb.execute(gdp, [gdp])

        open_prims = [prim for prim in gdp.iterPrims() if not prim.isClosed()]
        gdp.deletePrims(open_prims)

        return gdp

    def tesselate_geo_with_network(self, geo):
        """Takes an hou.Geometry and returns a tesselated version"""

        if self.tesselator is None:
            raise TypeError("Tesselator is None")
        self.tesselator.setCachedUserData("gdp", geo)
        self.tesselator.node("python").cook(force=True)
        gdp = self.tesselator.node("OUT").geometry().freeze()
        return gdp

    def create_tesselator(self):
        """Builds a SOP network for the tesselating geometry"""

        if hou.applicationVersion() >= HVER_17_5:
            return None

        # A network is created instead of a chain of Verbs because currently
        # the Convert SOP doesn't exist in Verb form.
        sopnet = hou.node("/out").createNode("sopnet")

        py_node = sopnet.createNode(
            "python", node_name="python", run_init_scripts=False
        )
        convert_node = sopnet.createNode(
            "convert", node_name="to_polys", run_init_scripts=False
        )
        divide_node = sopnet.createNode(
            "divide", node_name="triangulate", run_init_scripts=False
        )
        wrangler_node = sopnet.createNode(
            "attribwrangle", node_name="cull_open", run_init_scripts=False
        )
        out_node = sopnet.createNode("output", node_name="OUT", run_init_scripts=False)

        py_node.setUnloadFlag(True)
        convert_node.setUnloadFlag(True)
        out_node.setDisplayFlag(True)
        out_node.setRenderFlag(True)

        py_node.parm("python").set(self._tesslate_py)

        convert_node.parm("lodu").set(1)
        convert_node.parm("lodv").set(1)

        # Remove any primitives that are not closed as pbrt can not handle them
        wrangler_node.parm("class").set("primitive")
        wrangler_node.parm("snippet").set(
            'if (!primintrinsic(geoself(), "closed", @primnum)) '
            "removeprim(geoself(), @primnum, 1);"
        )

        convert_node.setFirstInput(py_node)
        divide_node.setFirstInput(convert_node)
        wrangler_node.setFirstInput(divide_node)
        out_node.setFirstInput(wrangler_node)

        return sopnet

    def remove_tesselator(self):
        """Tear down the previously created tesselator network"""
        if self.tesselator is None:
            return
        self.tesselator.destroyCachedUserData("gdp")
        self.tesselator.destroy()
        self.tesselator = None
        return


# Module global to hold the overall state of the export
scene_state = PBRTState()
