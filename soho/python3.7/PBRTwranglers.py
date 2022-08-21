import math
import collections

import hou
import soho
import sohog

import PBRTgeo
import PBRTinstancing

import PBRTapi as api

from PBRTstate import scene_state
from PBRTsoho import SohoPBRT
from PBRTnodes import PBRTParam, ParamSet, BaseNode
from PBRTshading import wrangle_shading_network

__all__ = [
    "wrangle_film",
    "wrangle_sampler",
    "wrangle_accelerator",
    "wrangle_integrator",
    "wrangle_options",
    "wrangle_filter",
    "wrangle_camera",
    "wrangle_light",
    "wrangle_preworld_medium",
    "wrangle_medium",
    "wrangle_obj",
]

ShutterRange = collections.namedtuple("ShutterRange", ["open", "close"])


def _apiclosure(api_call, *args, **kwargs):
    def api_func():
        return api_call(*args, **kwargs)

    return api_func


# This function is not being used currently but is a common pattern
# with other SOHO exporters
def get_wrangler(obj, now, style):  # pragma: no coverage
    wrangler = obj.getDefaultedString(style, now, [""])[0]
    wrangler = "%s-PBRT" % wrangler
    if style == "light_wrangler":
        wrangler = soho.LightWranglers.get(wrangler, None)
    elif style == "camera_wrangler":
        wrangler = soho.CameraWranglers.get(wrangler, None)
    elif style == "object_wrangler":
        wrangler = soho.ObjectWranglers.get(wrangler, None)
    else:
        wrangler = None  # Not supported at the current time

    if wrangler:
        wrangler = wrangler(obj, now, "PBRT")
    else:
        wrangler = None
    return wrangler


def get_transform(obj, now, invert=False, flipx=False, flipy=False, flipz=False):
    xform = []
    if not obj.evalFloat("space:world", now, xform):
        return None
    xform = hou.Matrix4(xform)
    if invert:
        xform = xform.inverted()
    x = -1 if flipx else 1
    y = -1 if flipy else 1
    z = -1 if flipz else 1
    xform = xform * hou.hmath.buildScale(x, y, z)
    return list(xform.asTuple())


def xform_to_api_srt(xform, scale=True, rotate=True, trans=True):
    xform = hou.Matrix4(xform)
    srt = xform.explode()
    if trans:
        api.Translate(*srt["translate"])
    if rotate:
        # NOTE, be wary of -180 to 180 flips
        rot = srt["rotate"]
        if rot.z():
            api.Rotate(rot[2], 0, 0, 1)
        if rot.y():
            api.Rotate(rot[1], 0, 1, 0)
        if rot.x():
            api.Rotate(rot[0], 1, 0, 0)
    if scale:
        api.Scale(*srt["scale"])
    return


def output_xform(
    obj,
    now,
    no_motionblur=False,
    invert=False,
    flipx=False,
    flipy=False,
    flipz=False,
    concat=False,
):
    if no_motionblur:
        shutter_range = None
    else:
        shutter_range = wrangle_motionblur(obj, now)

    api_call = api.Transform
    if concat:
        api_call = api.ConcatTransform

    if shutter_range is None:
        xform = get_transform(
            obj, now, invert=invert, flipx=flipx, flipy=flipy, flipz=flipz
        )
        api_call(xform)
        return
    api.ActiveTransform("StartTime")
    xform = get_transform(
        obj, shutter_range.open, invert=invert, flipx=flipx, flipy=flipy, flipz=flipz
    )
    api_call(xform)
    api.ActiveTransform("EndTime")
    xform = get_transform(
        obj, shutter_range.close, invert=invert, flipx=flipx, flipy=flipy, flipz=flipz
    )
    api_call(xform)
    api.ActiveTransform("All")
    return


def wrangle_node_parm(obj, parm_name, now):

    parm_selection = {parm_name: SohoPBRT(parm_name, "string", [""], False)}
    parms = obj.evaluate(parm_selection, now)
    if not parms:
        return None
    node_path = parms[parm_name].Value[0]
    if not node_path:
        return None
    return BaseNode(node_path)


def process_full_pt_instance_medium(instance_info, medium_type):
    gdp = instance_info.gdp

    if medium_type not in ("interior", "exterior"):
        return None, None

    medium_attrib_h = gdp.attribute("geo:point", "pbrt_" + medium_type)
    if medium_attrib_h < 0:
        return None, None

    medium = gdp.value(medium_attrib_h, instance_info.number)[0]

    # an empty string is valid here as it means no medium
    if medium == "":
        return medium, ParamSet()

    suffix = ":%s:%i" % (instance_info.source, instance_info.number)
    medium_node = BaseNode.from_node(medium)
    medium_node.path_suffix = suffix

    if not (medium_node and medium_node.directive == "medium"):
        return None, None

    medium_paramset = ParamSet(medium_node.paramset)

    # Look for attributes to create an override dictionary
    pt_attribs = gdp.globalValue("geo:pointattribs")
    overrides = {}
    for attrib in pt_attribs:
        if not attrib.startswith("{}_".format(medium_type)):
            continue
        if attrib in ("pbrt_interior", "pbrt_exterior"):
            continue
        parm_name = attrib.split("_", 1)[1]
        attrib_h = gdp.attribute("geo:point", attrib)
        if attrib_h < 0:
            continue
        val = gdp.value(attrib_h, instance_info.number)
        overrides[parm_name] = val
    if overrides:
        medium_paramset.update(medium_node.override_paramset(overrides))

    # We might be outputing a named medium even if its not going to be needed
    # as in the case of instancing volume prims
    api.MakeNamedMedium(
        medium_node.full_name, medium_node.directive_type, medium_paramset
    )

    return medium_node.full_name, medium_paramset


def process_full_pt_instance_material(instance_info):

    # The order of evaluation.
    #   1. shaders attached to the actual prims (handled in PBRTgeo.py)
    #   2. per point assignments on the instancer
    #   3. instancer's shop_materialpath
    #   4. object being instanced shop_materialpath
    #   5. nothing
    #   The choice between 3 and 4 is handled automatically by soho

    gdp = instance_info.gdp

    override_attrib_h = gdp.attribute("geo:point", "material_override")
    shop_attrib_h = gdp.attribute("geo:point", "shop_materialpath")

    if shop_attrib_h < 0:
        return False

    shop = gdp.value(shop_attrib_h, instance_info.number)[0]

    override_str = ""
    if override_attrib_h >= 0:
        override_str = gdp.value(override_attrib_h, instance_info.number)[0]

    # We can just reference a NamedMaterial since there are no overrides
    if not override_str:
        if shop in scene_state.shading_nodes:
            api.NamedMaterial(shop)
        else:
            # This shouldn't happen, if it does there is an coding mistake
            raise ValueError("Could not find shop in scene state")
        return True

    overrides = eval(override_str, {}, {})

    # override and shop should exist beyond this point
    # Fully expand shading network since there will be uniqueness
    suffix = ":%s:%i" % (instance_info.source, instance_info.number)
    # NOTE: If this becomes a bottleneck we could potentially cache nodes and params
    # similar to what we do in the PBRTgeo
    wrangle_shading_network(
        shop,
        use_named=False,
        exported_nodes=set(),
        name_suffix=suffix,
        overrides=overrides,
    )
    return True


def wrangle_motionblur(obj, now):
    mb_parms = [
        soho.SohoParm("allowmotionblur", "int", [0], False),
        soho.SohoParm("shutter", "float", [scene_state.shutter], False),
        soho.SohoParm("shutteroffset", "float", [None], False),
        soho.SohoParm("motionstyle", "string", ["trailing"], False),
    ]
    eval_mb_parms = obj.evaluate(mb_parms, now)
    if not eval_mb_parms[0].Value[0]:
        return None
    shutter = eval_mb_parms[1].Value[0] * scene_state.inv_fps
    offset = eval_mb_parms[2].Value[0]
    style = eval_mb_parms[3].Value[0]
    # This logic is in part from RIBmisc.py
    # NOTE: For pbrt output we will keep this limited to just shutter and
    #       shutteroffset, if the need arises we can add in the various
    #       scaling options etc.
    if style == "centered":
        delta = shutter * 0.5
    elif style == "leading":
        delta = shutter
    else:  # trailing
        delta = 0.0
    delta -= (offset - 1.0) * 0.5 * shutter
    start_time = now - delta
    end_time = start_time + shutter
    return ShutterRange(start_time, end_time)


def wrangle_film(obj, wrangler, now):

    node = wrangle_node_parm(obj, "film_node", now)
    if node is not None:
        return node.type_and_paramset

    paramset = ParamSet()

    parm_selection = {
        "filename": SohoPBRT("filename", "string", ["pbrt.exr"], False),
        "maxcomponentvalue": SohoPBRT("maxcomponentvalue", "float", [1e38], True),
        "diagonal": SohoPBRT("diagonal", "float", [35], True),
        "savefp16": SohoPBRT("savefp16", "bool", [1], True),
    }
    parms = obj.evaluate(parm_selection, now)
    for parm in parms.values():
        paramset.add(parm.to_pbrt())

    parm_selection = {
        "film": SohoPBRT("film", "string", ["rgb"], False),
        "res": SohoPBRT("res", "integer", [1280, 720], False),
    }
    parms = obj.evaluate(parm_selection, now)
    film_name = parms["film"].Value[0]
    paramset.add(PBRTParam("integer", "xresolution", parms["res"].Value[0]))
    paramset.add(PBRTParam("integer", "yresolution", parms["res"].Value[1]))

    crop_region = obj.getCameraCropWindow(wrangler, now)
    if crop_region != [0.0, 1.0, 0.0, 1.0]:
        paramset.add(PBRTParam("float", "cropwindow", crop_region))

    parm_selection = {
        "iso": SohoPBRT("iso", "float", [100], True),
        "whitebalance": SohoPBRT("whitebalance", "float", [0], True),
        "sensor": SohoPBRT("sensor", "string", ["cie1931"], True),
    }

    if film_name == "spectral":
        parm_selection["buckets"] = SohoPBRT("buckets", "integer", [16], True)

    parms = obj.evaluate(parm_selection, now)
    for parm in parms.values():
        paramset.add(parm.to_pbrt())

    return (film_name, paramset)


def wrangle_filter(obj, wrangler, now):

    node = wrangle_node_parm(obj, "filter_node", now)
    if node is not None:
        return node.type_and_paramset

    parm_selection = {
        "filter": SohoPBRT("filter", "string", ["gaussian"], False),
        "filter_radius": SohoPBRT("filter_radius", "float", [1.5, 1.5], False),
        "sigma": SohoPBRT("gauss_sigma", "float", [0.5], True, key="sigma"),
        "B": SohoPBRT("mitchell_B", "float", [0.333333], True, key="B"),
        "C": SohoPBRT("mitchell_C", "float", [0.333333], True, key="C"),
        "tau": SohoPBRT("sinc_tau", "float", [3], True, key="tau"),
    }
    parms = obj.evaluate(parm_selection, now)

    filter_name = parms["filter"].Value[0]
    paramset = ParamSet()
    xradius = parms["filter_radius"].Value[0]
    yradius = parms["filter_radius"].Value[1]
    paramset.add(PBRTParam("float", "xradius", xradius))
    paramset.add(PBRTParam("float", "yradius", yradius))

    if filter_name == "gaussian" and "sigma" in parms:
        paramset.add(parms["sigma"].to_pbrt())
    if filter_name == "mitchell" and "B" in parms:
        paramset.add(parms["B"].to_pbrt())
    if filter_name == "mitchell" and "C" in parms:
        paramset.add(parms["C"].to_pbrt())
    if filter_name == "sinc" and "tau" in parms:
        paramset.add(parms["tau"].to_pbrt())
    return (filter_name, paramset)


def wrangle_sampler(obj, wrangler, now):

    node = wrangle_node_parm(obj, "sampler_node", now)
    if node is not None:
        return node.type_and_paramset

    parm_selection = {
        "sampler": SohoPBRT("sampler", "string", ["pmj02bn"], False),
        "pixelsamples": SohoPBRT("pixelsamples", "integer", [16], False),
        "randomization": SohoPBRT("randomization", "string", ["fastowen"], True),
        "jitter": SohoPBRT("jitter", "bool", [1], True),
        "samples": SohoPBRT("samples", "integer", [4, 4], False),
    }
    parms = obj.evaluate(parm_selection, now)

    sampler_name = parms["sampler"].Value[0]
    paramset = ParamSet()

    if sampler_name == "stratified":
        xsamples = parms["samples"].Value[0]
        ysamples = parms["samples"].Value[1]
        paramset.add(PBRTParam("integer", "xsamples", xsamples))
        paramset.add(PBRTParam("integer", "ysamples", ysamples))
        if "jitter" in parms:
            paramset.add(parms["jitter"].to_pbrt())
    else:
        if (
            sampler_name in ("sobol", "paddedsobol", "zsobol", "halton")
            and "randomization" in parms
        ):
            # NOTE: If the halton sampler is picked, it is not compatible with the
            # randomization "fastowen".
            paramset.add(parms["randomization"].to_pbrt())
        paramset.add(parms["pixelsamples"].to_pbrt())

    return (sampler_name, paramset)


def wrangle_options(obj, wrangler, now):
    parm_selection = {
        "disabletexturefiltering": SohoPBRT(
            "disabletexturefiltering", "bool", [False], True
        ),
        "disablepixeljitter": SohoPBRT("disablepixeljitter", "bool", [False], True),
        "disablewavelengthjitter": SohoPBRT(
            "disablewavelengthjitter", "bool", [False], True
        ),
        "msereferenceimage": SohoPBRT("msereferenceimage", "string", [""], True),
        "msereferenceout": SohoPBRT("msereferenceout", "string", [""], True),
        # NOTE:
        # The PBRT default is "cameraworld" but this can cause instancing interpolation
        # issues. For now we'll use the PBRT default and expect the user to switch to
        # world, but depending on the resolution of
        # https://github.com/mmp/pbrt-v4/issues/206
        # we may need to default this to "world"
        "rendercoordsys": SohoPBRT("rendercoordsys", "string", ["cameraworld"], True),
        "seed": SohoPBRT("seed", "integer", [0], True),
        "displacementedgescale": SohoPBRT(
            "displacementedgescale", "float", [1.0], True
        ),
        "forcediffuse": SohoPBRT("forcediffuse", "bool", [False], True),
        "pixelstats": SohoPBRT("pixelstats", "bool", [False], True),
        "wavefront": SohoPBRT("wavefront", "bool", [False], True),
    }
    parms = obj.evaluate(parm_selection, now)

    for parm in parms:
        yield (parm, parms[parm].to_pbrt())


def wrangle_integrator(obj, wrangler, now):

    node = wrangle_node_parm(obj, "integrator_node", now)
    if node is not None:
        return node.type_and_paramset

    parm_selection = {
        "integrator": SohoPBRT("integrator", "string", ["path"], False),
        "maxdepth": SohoPBRT("maxdepth", "integer", [5], False),
        "regularize": SohoPBRT("regularize", "bool", [False], True),
        "lightsampler": SohoPBRT("lightsampler", "string", ["bvh"], True),
        "visualizestrategies": SohoPBRT("visualizestrategies", "bool", [False], True),
        "visualizeweights": SohoPBRT("visualizeweights", "bool", [False], True),
        "iterations": SohoPBRT("iterations", "integer", [64], True),
        "photonsperiteration": SohoPBRT("photonsperiteration", "integer", [-1], True),
        "radius": SohoPBRT("radius", "float", [1], True),
        "bootstrapsamples": SohoPBRT("bootstrapsamples", "integer", [100000], True),
        "chains": SohoPBRT("chains", "integer", [1000], True),
        "mutationsperpixel": SohoPBRT("mutationsperpixel", "integer", [100], True),
        "largestepprobability": SohoPBRT("largestepprobability", "float", [0.3], True),
        "sigma": SohoPBRT("sigma", "float", [0.01], True),
        "maxdistance": SohoPBRT("maxdistance", "float", ["1e38"], True),
        "cossample": SohoPBRT("cossample", "bool", [True], True),
        "samplelights": SohoPBRT("samplelights", "bool", [True], True),
        "samplebsdf": SohoPBRT("samplebsdf", "bool", [True], True),
    }

    integrator_parms = {
        "ambientocclusion": ["maxdistance", "cossample"],
        "path": ["maxdepth", "regularize", "lightsampler"],
        "bdpt": ["maxdepth", "regularize", "visualizestrategies", "visualizeweights"],
        "mlt": [
            "maxdepth",
            "bootstrapsamples",
            "chains",
            "mutationsperpixel",
            "largestepprobability",
            "sigma",
            "regularize",
        ],
        "sppm": [
            "maxdepth",
            "iterations",
            "photonsperiteration",
            "radius",
            "regularize",
        ],
        "lightpath": ["maxdepth"],
        "randomwalk": ["maxdepth"],
        "simplepath": ["maxdepth", "samplelights", "samplebsdf"],
        "simplevolpath": ["maxdepth"],
        "volpath": ["maxdepth", "lightsampler", "regularize"],
    }
    parms = obj.evaluate(parm_selection, now)

    integrator_name = parms["integrator"].Value[0]
    paramset = ParamSet()
    for parm_name in integrator_parms[integrator_name]:
        if parm_name not in parms:
            continue
        paramset.add(parms[parm_name].to_pbrt())

    return (integrator_name, paramset)


def wrangle_accelerator(obj, wrangler, now):

    node = wrangle_node_parm(obj, "accelerator_node", now)
    if node is not None:
        return node.type_and_paramset

    parm_selection = {"accelerator": SohoPBRT("accelerator", "string", ["bvh"], False)}
    parms = obj.evaluate(parm_selection, now)
    accelerator_name = parms["accelerator"].Value[0]

    if accelerator_name == "bvh":
        parm_selection = {
            "maxnodeprims": SohoPBRT("maxnodeprims", "integer", [4], True),
            "splitmethod": SohoPBRT("splitmethod", "string", ["sah"], True),
        }
    else:
        parm_selection = {
            "intersectcost": SohoPBRT("intersectcost", "integer", [80], True),
            "traversalcostcost": SohoPBRT("traversalcost", "integer", [1], True),
            "emptybonus": SohoPBRT("emptybonus", "float", [0.2], True),
            "maxprims": SohoPBRT("maxprims", "integer", [1], True),
            "kdtree_maxdepth": SohoPBRT(
                "kdtree_maxdepth", "integer", [1], True, key="maxdepth"
            ),
        }
    parms = obj.evaluate(parm_selection, now)

    paramset = ParamSet()

    for parm in parms:
        paramset.add(parms[parm].to_pbrt())

    return (accelerator_name, paramset)


def output_cam_xform(obj, projection, now):
    # NOTE: Initial tests show pbrt has problems when motion blur xforms
    #       are applied to the camera (outside the World block)
    if projection in ("perspective", "orthographic", "realistic"):
        output_xform(obj, now, no_motionblur=True, invert=True, flipz=True)
    elif projection in ("spherical",):
        api.Rotate(-180, 0, 1, 0)
        output_xform(obj, now, no_motionblur=True, invert=True, flipx=True, flipz=True)
    return


def wrangle_camera(obj, wrangler, now):

    node = wrangle_node_parm(obj, "camera_node", now)
    if node is not None:
        output_cam_xform(obj, node.directive_type, now)
        return node.type_and_paramset

    paramset = ParamSet()

    window = obj.getCameraScreenWindow(wrangler, now)
    parm_selection = {
        "projection": SohoPBRT("projection", "string", ["perspective"], False),
        "focal": SohoPBRT("focal", "float", [50], False),
        "focalunits": SohoPBRT("focalunits", "string", ["mm"], False),
        "aperture": SohoPBRT("aperture", "float", [41.4214], False),
        "orthowidth": SohoPBRT("orthowidth", "float", [2], False),
        "res": SohoPBRT("res", "integer", [1280, 720], False),
        "aspect": SohoPBRT("aspect", "float", [1], False),
        "fstop": SohoPBRT("fstop", "float", [5.6], False),
        "focaldistance": SohoPBRT("focus", "float", [5], False, key="focaldistance"),
        "pbrt_dof": SohoPBRT("pbrt_dof", "integer", [0], False),
    }

    parms = obj.evaluate(parm_selection, now)
    aspect = parms["aspect"].Value[0]
    aspectfix = aspect * float(parms["res"].Value[0]) / float(parms["res"].Value[1])

    projection = parms["projection"].Value[0]

    if parms["pbrt_dof"].Value[0]:
        paramset.add(parms["focaldistance"].to_pbrt())
        # to convert from f-stop to lens radius
        # FStop = FocalLength / (Radius * 2)
        # Radius = FocalLength/(FStop * 2)
        focal = parms["focal"].Value[0]
        fstop = parms["fstop"].Value[0]
        units = parms["focalunits"].Value[0]
        focal = soho.houdiniUnitLength(focal, units)
        lens_radius = focal / (fstop * 2.0)
        paramset.add(PBRTParam("float", "lensradius", lens_radius))

    if projection == "perspective":
        projection_name = "perspective"

        focal = parms["focal"].Value[0]
        aperture = parms["aperture"].Value[0]
        fov = 2.0 * focal / aperture
        fov = 2.0 * math.degrees(math.atan2(1.0, fov))
        paramset.add(PBRTParam("float", "fov", [fov]))

        screen = [
            (window[0] - 0.5) * 2.0,
            (window[1] - 0.5) * 2.0,
            (window[2] - 0.5) * 2.0 / aspectfix,
            (window[3] - 0.5) * 2.0 / aspectfix,
        ]
        paramset.add(PBRTParam("float", "screenwindow", screen))

    elif projection == "ortho":
        projection_name = "orthographic"

        width = parms["orthowidth"].Value[0]
        screen = [
            (window[0] - 0.5) * width,
            (window[1] - 0.5) * width,
            (window[2] - 0.5) * width / aspectfix,
            (window[3] - 0.5) * width / aspectfix,
        ]
        paramset.add(PBRTParam("float", "screenwindow", screen))

    elif projection == "sphere":
        projection_name = "spherical"
    else:
        soho.error("Camera projection setting of %s not supported by PBRT" % projection)

    output_cam_xform(obj, projection_name, now)

    return (projection_name, paramset)


def _to_light_scale(parms):
    """Converts light_intensity, light_exposure to a single scale value"""
    intensity = parms["light_intensity"].Value[0]
    exposure = parms["light_exposure"].Value[0]
    scale = intensity * (2.0**exposure)
    return PBRTParam("float", "scale", [scale])


def _light_api_wrapper(wrangler_light_type, wrangler_paramset, node):
    if node is not None:
        ltype = node.directive_type
        paramset = node.paramset
        is_arealight = bool(node.directive == "arealight")
    else:
        ltype = wrangler_light_type
        paramset = wrangler_paramset
        is_arealight = bool(ltype == "diffuse")

    if is_arealight:
        api.AreaLightSource(ltype, paramset)
    else:
        api.LightSource(ltype, paramset)


def _portal_helper(now, portal):
    gdp = sohog.SohoGeometry(portal, now)
    if gdp.Handle < 0:
        api.Comment("No geometry available, skipping")
        return None
    pt_count = gdp.globalValue("geo:pointcount")
    if pt_count < 4:
        return None
    P_h = gdp.attribute("geo:point", "P")
    portal_pts = []
    for i in range(4):
        portal_pts.append(gdp.value(P_h, i))
    return portal_pts


def wrangle_light(light, wrangler, now):

    # NOTE: Lights do not support motion blur so we disable it when
    #       outputs the xforms

    node = wrangle_node_parm(light, "light_node", now)

    # We skip the light_color if its at default so we avoid setting rgb values
    # if at all possible, that way we get a constant spectrum instead
    parm_selection = {
        "light_wrangler": SohoPBRT("light_wrangler", "string", [""], False),
        "light_color": SohoPBRT("light_color", "float", [1, 1, 1], True),
        "light_intensity": SohoPBRT("light_intensity", "float", [1], False),
        "light_exposure": SohoPBRT("light_exposure", "float", [0], False),
    }
    parms = light.evaluate(parm_selection, now)
    light_wrangler = parms["light_wrangler"].Value[0]

    exterior = light.wrangleString(wrangler, "pbrt_exterior", now, [None])[0]
    exterior = wrangle_medium(exterior)
    if exterior:
        api.MediumInterface("", exterior)
        print()

    paramset = ParamSet()
    paramset.add(_to_light_scale(parms))

    if light_wrangler == "HoudiniEnvLight":
        env_map = []
        light.evalString("env_map", now, env_map)
        # evalString will return [""] if the parm exists yet at its default
        env_map = env_map[0] if env_map else ""
        if env_map:
            paramset.add(PBRTParam("string", "filename", env_map))
        elif "light_color" in parms:
            paramset.add(PBRTParam("rgb", "L", parms["light_color"].Value))

        portal = light.wrangleString(wrangler, "env_portal", now, [""])[0]
        portal_enabled = light.wrangleInt(wrangler, "env_portalenable", now, [0])[0]
        if portal_enabled and portal:
            portal_pts = _portal_helper(now, portal)
            if portal_pts is not None:
                # TODO pbrt-v4 we may need to invert the Houdini -> PBRT xform
                paramset.add(PBRTParam("point", "portal", portal_pts))

        output_xform(light, now, no_motionblur=True)
        api.Scale(1, 1, -1)
        api.Rotate(90, 0, 0, 1)
        api.Rotate(90, 0, 1, 0)
        _light_api_wrapper("infinite", paramset, node)
        return
    elif light_wrangler != "HoudiniLight":
        api.Comment("This light type, %s, is unsupported" % light_wrangler)
        return

    # We are dealing with a standard HoudiniLight type.

    light_type = light.wrangleString(wrangler, "light_type", now, ["point"])[0]

    if light_type in ("sphere", "disk", "grid", "tube", "geo"):

        single_sided = light.wrangleInt(wrangler, "singlesided", now, [0])[0]
        reverse = light.wrangleInt(wrangler, "reverse", now, [0])[0]
        visible = light.wrangleInt(wrangler, "light_contribprimary", now, [0])[0]
        size = light.wrangleFloat(wrangler, "areasize", now, [1, 1])
        paramset.add(PBRTParam("bool", "twosided", [not single_sided]))

        texmap = light.wrangleString(wrangler, "light_texture", now, [""])[0]
        if texmap:
            paramset.add(PBRTParam("string", "filename", texmap))
        elif "light_color" in parms:
            paramset.add(PBRTParam("rgb", "L", parms["light_color"].Value))

        # TODO, Possibly get the xform's scale and scale the geo, not the light.
        #       (for example, further multiplying down the radius)
        xform = get_transform(light, now)
        xform_to_api_srt(xform, scale=False)

        _light_api_wrapper("diffuse", paramset, node)

        api.AttributeBegin()

        if single_sided and reverse:
            api.ReverseOrientation()

        shape_paramset = ParamSet()
        if not visible:
            shape_paramset.add(PBRTParam("float", "alpha", 0.0))

        # PBRT only supports uniform scales for non-mesh area lights
        # this is in part due to explicit light's area scaling factor.
        if light_type in ("grid", "geo"):
            api.Scale(size[0], size[1], size[0])

        if light_type == "sphere":
            # NOTE:
            # To match the UVs we need a api.Scale(1, 1, -1)
            # However doing this screws up the direction of emission.
            # When rendering as one sided, the emissive side will be the opposite
            # side from which is used to illuminate. Unfortunately an
            # api.ReverseOrientation() does not fix this.

            # We apply the scale to the radius instead of using a api.Scale
            shape_paramset.add(PBRTParam("float", "radius", 0.5 * size[0]))
            api.Shape("sphere", shape_paramset)
        elif light_type == "tube":
            api.Rotate(90, 0, 1, 0)
            api.Rotate(90, 0, 0, 1)
            # NOTE:
            # To match UVs we need a api.Scale(1, 1, -1)
            # see note above about spheres.
            shape_paramset.add(PBRTParam("float", "radius", 0.075 * size[1]))
            shape_paramset.add(PBRTParam("float", "zmin", -0.5 * size[0]))
            shape_paramset.add(PBRTParam("float", "zmax", 0.5 * size[0]))
            api.Shape("cylinder", shape_paramset)
        elif light_type == "disk":
            # NOTE this should match mantra now, unlike in pbrt-v3
            api.Scale(-1, 1, -1)
            shape_paramset.add(PBRTParam("float", "radius", 0.5 * size[0]))
            api.Shape("disk", shape_paramset)
        elif light_type == "grid":
            api.ReverseOrientation()
            shape_paramset.add(
                PBRTParam(
                    "point",
                    "P",
                    [-0.5, -0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, 0.5, 0.5, 0],
                )
            )
            api.Shape("bilinearmesh", shape_paramset)
        elif light_type == "geo":
            areageo_parm = hou.node(light.getName()).parm("areageometry")
            if not areageo_parm:
                api.Comment('No "areageometry" parm on light')
                return
            area_geo_node = areageo_parm.evalAsNode()
            if not area_geo_node:
                api.Comment("Skipping, no geometry object specified")
                return
            obj = soho.getObject(area_geo_node.path())
            api.Comment("Light geo from %s" % obj.getName())
            # TODO: The area light scale ('areasize') happens *after* the wrangle_obj's
            #       xform when 'intothisobject' is enabled.

            # TODO: the Light visiblity paramset ("alpha") can't be easily passed
            #       with this current interface. It can be worked aroudn by setting
            #       the referenced object's "alpha" property
            into_this_obj = light.wrangleInt(wrangler, "intothisobject", now, [0])[0]
            ignore_xform = not into_this_obj
            wrangle_obj(obj, None, now, ignore_xform=ignore_xform)

        api.AttributeEnd()

        return

    cone_enable = light.wrangleInt(wrangler, "coneenable", now, [0])[0]
    projmap = light.wrangleString(wrangler, "projmap", now, [""])[0]
    areamap = light.wrangleString(wrangler, "areamap", now, [""])[0]

    api_calls = []
    api_calls.append(_apiclosure(output_xform, light, now, no_motionblur=True))
    api_calls.append(_apiclosure(api.Scale, 1, 1, -1))
    api_calls.append(_apiclosure(api.Scale, 1, -1, 1))

    if light_type == "point":
        if areamap:
            light_name = "goniometric"
            if "light_color" in parms:
                paramset.add(PBRTParam("rgb", "I", parms["light_color"].Value))
            paramset.add(PBRTParam("string", "filename", [areamap]))
            api_calls = []
            api_calls.append(_apiclosure(output_xform, light, now, no_motionblur=True))
            api_calls.append(_apiclosure(api.Scale, 1, -1, 1))
            api_calls.append(_apiclosure(api.Rotate, 90, 0, 1, 0))
        elif not cone_enable:
            light_name = "point"
            if "light_color" in parms:
                paramset.add(PBRTParam("rgb", "I", parms["light_color"].Value))
        elif projmap:
            light_name = "projection"
            coneangle = light.wrangleFloat(wrangler, "coneangle", now, [45])[0]
            paramset.add(PBRTParam("float", "fov", [coneangle]))
            paramset.add(PBRTParam("string", "filename", [projmap]))
        else:
            light_name = "spot"
            if "light_color" in parms:
                paramset.add(PBRTParam("rgb", "I", parms["light_color"].Value))
            conedelta = light.wrangleFloat(wrangler, "conedelta", now, [10])[0]
            coneangle = light.wrangleFloat(wrangler, "coneangle", now, [45])[0]
            coneangle *= 0.5
            coneangle += conedelta
            paramset.add(PBRTParam("float", "coneangle", [coneangle]))
            paramset.add(PBRTParam("float", "conedeltaangle", [conedelta]))
    elif light_type == "distant":
        light_name = light_type
        if "light_color" in parms:
            paramset.add(PBRTParam("rgb", "L", parms["light_color"].Value))
    else:
        api.Comment("Light Type, %s, not supported" % light_type)
        return

    for api_call in api_calls:
        api_call()
    _light_api_wrapper(light_name, paramset, node)

    return


def wrangle_preworld_medium(obj, wrangler, now):
    """Output a NamedMedium from the input oppath"""

    # Due to PBRT's scene description we can't use the standard wrangle_medium
    # when declaring a medium and its transform/colorspace when attached to a
    # camera. This is because we don't have AttributeBegin/End blocks to pop
    # the stack, so when we declare the transform for the medium we need to so
    # with respect to being in the camera's coordinate system. We'll extract
    # the translates from the camera, to establish a pivot.
    medium = obj.wrangleString(wrangler, "pbrt_exterior", now, [None])[0]

    if not medium:
        return None
    if medium in scene_state.medium_nodes:
        return None
    scene_state.medium_nodes.add(medium)

    medium_vop = BaseNode.from_node(medium)
    if medium_vop is None:
        return None
    if medium_vop.directive != "medium":
        return None

    coord_sys = medium_vop.coord_sys
    if coord_sys:
        cam_xform = hou.Matrix4(get_transform(obj, now, invert=False, flipz=False))
        cam_pivot = cam_xform.extractTranslates()
        cam_pivot = hou.hmath.buildTranslate(cam_pivot).inverted()
        xform = hou.Matrix4(coord_sys)
        xform *= cam_pivot

        api.Transform(xform.asTuple())

    colorspace = medium_vop.colorspace
    if colorspace:
        api.ColorSpace(colorspace)

    api.MakeNamedMedium(medium_vop.path, medium_vop.directive_type, medium_vop.paramset)
    api.Identity()
    # Restore Colorspace if one was set on the Medium
    if colorspace:
        scene_cs = []
        if obj.evalString("pbrt_colorspace", now, scene_cs):
            scene_cs = scene_cs[0]
        else:
            scene_cs = "srgb"
        if scene_cs != colorspace:
            api.ColorSpace(scene_cs)

    return medium_vop.path


def wrangle_medium(medium):
    """Output a NamedMedium from the input oppath"""
    if not medium:
        return None
    if medium in scene_state.medium_nodes:
        return None
    scene_state.medium_nodes.add(medium)

    medium_vop = BaseNode.from_node(medium)
    if medium_vop is None:
        return None
    if medium_vop.directive != "medium":
        return None

    with api.AttributeBlock():

        coord_sys = medium_vop.coord_sys
        if coord_sys:
            api.Transform(coord_sys)

        colorspace = medium_vop.colorspace
        if colorspace:
            api.ColorSpace(colorspace)

        api.MakeNamedMedium(
            medium_vop.path, medium_vop.directive_type, medium_vop.paramset
        )

    return medium_vop.path


def wrangle_obj(obj, wrangler, now, ignore_xform=False, concat_xform=False):

    ptinstance = []
    has_ptinstance = obj.evalInt("ptinstance", now, ptinstance)

    if not ignore_xform:
        output_xform(obj, now, concat=concat_xform)

    if has_ptinstance and ptinstance[0] == 2:
        shutter_times = wrangle_motionblur(obj, now)
        ptmotionblur = []
        has_ptmotionblur = obj.evalString("ptmotionblur", now, ptmotionblur)
        if (
            shutter_times is not None
            and has_ptmotionblur
            and ptmotionblur[0] == "deform"
        ):
            times = (shutter_times.open, shutter_times.close)
        else:
            times = (now,)
        # This is "fast instancing", "full instancing" results in Soho outputing
        # actual objects which independently need to be wrangled.
        PBRTinstancing.wrangle_fast_instances(obj, times)
        return

    wrangle_geo(obj, wrangler, now)
    return


def wrangle_geo(obj, wrangler, now):
    parm_selection = {
        "object:soppath": SohoPBRT("object:soppath", "string", [""], skipdefault=False),
        "ptinstance": SohoPBRT("ptinstance", "integer", [0], skipdefault=False),
        # NOTE: In order for shop_materialpath to evaluate correctly when using
        #       (full) instancing shop_materialpath needs to be a 'shaderhandle'
        #       and not a 'string'
        # NOTE: However this does not seem to apply to shop_materialpaths on the
        #       instance points and has to be done manually
        "shop_materialpath": SohoPBRT(
            "shop_materialpath", "shaderhandle", skipdefault=False
        ),
        "soho_precision": SohoPBRT("soho_precision", "integer", [9], False),
        "pbrt_rendersubd": SohoPBRT("pbrt_rendersubd", "bool", [False], False),
        "pbrt_subdlevels": SohoPBRT(
            "pbrt_subdlevels", "integer", [3], False, key="levels"
        ),
        "pbrt_computeN": SohoPBRT("pbrt_computeN", "bool", [True], False),
        "pbrt_reverseorientation": SohoPBRT(
            "pbrt_reverseorientation", "bool", [False], True
        ),
        "pbrt_matchhoudiniuv": SohoPBRT("pbrt_matchhoudiniuv", "bool", [True], False),
        # The combination of None as a default as well as ignore defaults being False
        # is important. 'None' implying the parm is missing and not available,
        # and '' meaning a vacuum medium.
        # We can't ignore defaults since a default might be the only way to set a
        # medium back to a vacuum.
        "pbrt_interior": SohoPBRT("pbrt_interior", "string", [None], False),
        "pbrt_exterior": SohoPBRT("pbrt_exterior", "string", [None], False),
        "pbrt_ignorevolumes": SohoPBRT("pbrt_ignorevolumes", "bool", [False], True),
        "pbrt_ignorematerials": SohoPBRT("pbrt_ignorematerials", "bool", [False], True),
        "pbrt_splitdepth": SohoPBRT(
            "pbrt_splitdepth", "integer", [3], True, key="splitdepth"
        ),
        "pbrt_emissionfilename": SohoPBRT(
            "pbrt_emissionfilename", "string", [""], True
        ),
        "pbrt_curvetype": SohoPBRT("pbrt_curvetype", "string", ["flat"], True),
        "pbrt_include": SohoPBRT("pbrt_include", "string", [""], False),
        "pbrt_import": SohoPBRT("pbrt_import", "string", [""], False),
        "pbrt_alpha_texture": SohoPBRT(
            "pbrt_alpha_texture", "string", [""], False, key="alpha"
        ),
        "pbrt_allow_geofiles": soho.SohoParm(
            "pbrt_allow_geofiles", "integer", [1], False
        ),
        "pbrt_geo_location": soho.SohoParm(
            "pbrt_geo_location", "string", ["geometry"], False
        ),
        "pbrt_geofile_threshold": soho.SohoParm(
            "pbrt_geofile_threshold", "integer", [10000], False
        ),
        "pbrt_renderpoints": soho.SohoParm("pbrt_renderpoints", "bool", [False], False),
    }
    properties = obj.evaluate(parm_selection, now)

    if "shop_materialpath" not in properties:
        shop = ""
    else:
        shop = properties["shop_materialpath"].Value[0]

    # NOTE: Having to track down shop_materialpaths does not seem to be a requirement
    #       with Mantra or RenderMan. Either its because I'm missing some
    #       logic/initialization either in Soho or in the Shading HDAs. Or there is
    #       some hardcoding in the Houdini libs that know how to translate
    #       shop_materialpath point aassignments to shaders directly through a
    #       SohoParm. Until that is figured out, we'll have to do it manually.

    interior = None
    exterior = None
    if "pbrt_interior" in properties:
        interior = properties["pbrt_interior"].Value[0]
    if "pbrt_exterior" in properties:
        exterior = properties["pbrt_exterior"].Value[0]

    if "pbrt_reverseorientation" in properties:
        if properties["pbrt_reverseorientation"].Value[0]:
            api.ReverseOrientation()

    pt_shop_found = False
    if properties["ptinstance"].Value[0] == 1:
        instance_info = PBRTinstancing.get_full_instance_info(obj, now)
        properties[".instance_info"] = instance_info
        if instance_info is not None:
            pt_shop_found = process_full_pt_instance_material(instance_info)
            interior, interior_paramset = process_full_pt_instance_medium(
                instance_info, "interior"
            )
            exterior, exterior_paramset = process_full_pt_instance_medium(
                instance_info, "exterior"
            )
            if interior_paramset is not None:
                properties[".interior_overrides"] = interior_paramset

    # If we found a point shop, don't output the default one here.
    if shop in scene_state.shading_nodes and not pt_shop_found:
        api.NamedMaterial(shop)

    # We only output a MediumInterface if one or both of the parms exist
    if interior is not None or exterior is not None:
        interior = "" if interior is None else interior
        exterior = "" if exterior is None else exterior
        api.MediumInterface(interior, exterior)

    alpha_tex = properties["alpha"].Value[0]
    alpha_node = BaseNode.from_node(alpha_tex)
    if (
        alpha_node
        and alpha_node.directive == "texture"
        and alpha_node.output_type == "float"
    ):
        if alpha_node.path not in scene_state.shading_nodes:
            suffix = ":%s" % obj.getName()
            alpha_tex = "%s%s" % (alpha_tex, suffix)
            properties["alpha"].Value[0] = alpha_tex
            wrangle_shading_network(
                alpha_node.path, name_suffix=suffix, exported_nodes=set()
            )
    else:
        # If the passed in alpha_texture wasn't valid, clear it so we don't add
        # it to the geometry
        if alpha_tex:
            api.Comment("%s is an invalid float texture" % alpha_tex)
        properties["alpha"].Value[0] = ""

    if properties["pbrt_import"].Value[0]:
        # If we have included a file, skip output any geo.
        api.Import(properties["pbrt_import"].Value[0])
        return

    if properties["pbrt_include"].Value[0]:
        # If we have included a file, skip output any geo.
        api.Include(properties["pbrt_include"].Value[0])
        return

    soppath = properties["object:soppath"].Value[0]
    if not soppath:
        api.Comment("Can not find soppath for object")
        return

    shutter_times = wrangle_motionblur(obj, now)
    if shutter_times is not None:
        times = (shutter_times.open, shutter_times.close)
    else:
        times = (now,)

    if properties["pbrt_renderpoints"].Value[0]:
        PBRTgeo.output_pts(soppath, times, properties)
    else:
        PBRTgeo.output_geo(soppath, now, properties)

    return
