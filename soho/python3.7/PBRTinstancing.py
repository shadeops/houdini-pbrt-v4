import collections

import hou
import soho
from sohog import SohoGeometry

import PBRTapi as api

_FullInstance = collections.namedtuple(
    "_FullInstance", ["instance", "source", "number", "gdp"]
)


def get_full_instance_info(obj, now):
    tokens = obj.getName().split(":")
    if len(tokens) != 3:
        return None
    instancer_obj = soho.getObject(tokens[1])
    instancer_sop = []
    if not instancer_obj.evalString("object:soppath", now, instancer_sop):
        return None
    instancer_sop = instancer_sop[0]
    gdp = SohoGeometry(instancer_sop, now)
    if gdp is None:
        return None
    return _FullInstance(tokens[0], tokens[1], int(tokens[2]), gdp)


def find_referenced_instances(obj):
    """Find and list any used instances in a Soho Object"""

    # We will be using a hou.Node instead of a Soho Object for this
    # so we can just query the strings directly instead of having to
    # iterate over all the points.

    obj_node = hou.node(obj.getName())
    if not obj_node:
        return

    sop_node = obj_node.renderNode()
    if sop_node is None:
        return

    geo = sop_node.geometry()
    if geo is None:
        return

    # Get the full path to any point instance geos
    instance_attrib = geo.findPointAttrib("instance")
    if (
        instance_attrib is not None
        and instance_attrib.dataType() == hou.attribData.String
    ):
        for instance_str in instance_attrib.strings():
            instance_obj = sop_node.node(instance_str)
            if instance_obj:
                yield instance_obj.path()

    # Get the object's instancepath as well
    instancepath_parm = obj_node.parm("instancepath")
    if instancepath_parm:
        instance_obj = instancepath_parm.evalAsNode()
        if instance_obj:
            yield instance_obj.path()


def wrangle_fast_instances(obj, times):
    """Output instanced geoemtry defined by fast instancing"""

    # NOTE: Homogenous volumes work when applied to a ObjectBegin/End however
    #       Heterogenous volumes do not. The p0 p1 params aren't being
    #       transformed properly by the instance's CTM.

    if len(times) == 2:
        now, close = times
    else:
        now = times[0]
        close = None

    soppath = []
    if not obj.evalString("object:soppath", now, soppath):
        api.Comment("Can not find soppath for object")
        return
    sop = soppath[0]

    obj_node = hou.node(obj.getName())
    sop_node = hou.node(sop)
    if obj_node is None or sop_node is None:
        api.Comment("Can not resolve obj or geo")
        return

    # Exit out quick if we can't fetch the proper instance attribs.
    geo = SohoGeometry(sop, now)
    if geo.Handle < 0:
        api.Comment("No geometry available, skipping")
        return

    num_pts = geo.globalValue("geo:pointcount")[0]
    if not num_pts:
        api.Comment("No points, skipping")
        return

    pt_attribs = (
        "geo:pointxform",
        "instance",
        # NOTE: Materials can not be applied to ObjectInstances
        # ( or setting material params (overrides) for that matter
        # See Excersise B.2 in 'The Book'
        # same applies for medium interfaces as well.
        # Applying them to the ObjectInstances does nothing
        # works on the base instance defintion
        # 'shop_materialpath',
        # 'material_override',
    )

    instancepath = []
    obj.evalString("instancepath", now, instancepath)
    instance_node = obj_node.node(instancepath[0])
    if instance_node is not None:
        default_instance_geo = instance_node.path()
    else:
        default_instance_geo = ""

    pt_attrib_map = {}
    for attrib in pt_attribs:
        attrib_h = geo.attribute("geo:point", attrib)
        if attrib_h >= 0:
            pt_attrib_map[attrib] = attrib_h

    if "geo:pointxform" not in pt_attrib_map:
        api.Comment("Can not find instance xform attribs, skipping")
        return

    if close is not None:
        geo_1 = SohoGeometry(sop, close)
        if geo_1.Handle < 0:
            api.Comment("No geometry at shutter close available, skipping")
            return

        num_pts_1 = geo_1.globalValue("geo:pointcount")[0]
        if not num_pts_1:
            api.Comment("No points at shutter close, skipping")
            return

        if num_pts != num_pts_1:
            api.Comment("Point count mismatch between shutter open and close, skipping")
            return

        pointxform_1_h = geo_1.attribute("geo:point", "geo:pointxform")
        if pointxform_1_h < 0:
            api.Comment(
                "Can not find instance xform attribs in shutter close, skipping"
            )
            return

    for pt in range(num_pts):
        instance_geo = default_instance_geo
        if "instance" in pt_attrib_map:
            pt_instance_geo = geo.value(pt_attrib_map["instance"], pt)[0]
            pt_instance_node = sop_node.node(pt_instance_geo)
            if pt_instance_node is not None:
                instance_geo = pt_instance_node.path()

        if not instance_geo:
            continue

        with api.AttributeBlock():
            api.Comment("%s:%i" % (sop, pt))
            xform = geo.value(pt_attrib_map["geo:pointxform"], pt)
            if close is None:
                api.ConcatTransform(xform)
            else:
                api.ActiveTransform("StartTime")
                api.ConcatTransform(xform)
                api.ActiveTransform("EndTime")
                xform = geo_1.value(pointxform_1_h, pt)
                api.ConcatTransform(xform)
                api.ActiveTransform("All")
            api.ObjectInstance(instance_geo)
    return
