from __future__ import print_function, division, absolute_import

import os
import time

import soho
from sohog import SohoGeometry

import PBRTapi as api
from PBRTwranglers import *  # noqa: F403
from PBRTinstancing import find_referenced_instances, get_full_instance_info
from PBRTshading import wrangle_shading_network
from PBRTstate import scene_state

# Ignore the various linting errors due to the import *
# flake8: noqa: F405


def output_materials(obj, wrangler, now, skip_included=False):
    """Output Materials for an object

    The shop_materialpath parameter and shop_materialpath prim attribute
    are both checked for output.
    """
    # We use a shaderhandle instead of a string so Soho instances are properly
    # resolved when Full Instancing is used.
    parms = {
        "shop_materialpath": soho.SohoParm(
            "shop_materialpath", "shaderhandle", skipdefault=False
        ),
        "pbrt_include": soho.SohoParm("pbrt_include", "string", [""], skipdefault=True),
        "pbrt_import": soho.SohoParm("pbrt_import", "string", [""], skipdefault=True),
    }

    eval_parms = obj.evaluate(parms, now)

    if "shop_materialpath" in eval_parms:
        shop = eval_parms["shop_materialpath"].Value[0]
        if shop:
            wrangle_shading_network(shop)

    # If this object uses either an include or import statement then we assume internal
    # material definitions are already resolved through other means.
    if skip_included and ("pbrt_include" in eval_parms or "pbrt_import" in eval_parms):
        return

    soppath = []
    if not obj.evalString("object:soppath", now, soppath):
        return

    soppath = soppath[0]

    gdp = SohoGeometry(soppath, now)
    global_material = gdp.globalValue("shop_materialpath")
    if global_material is not None:
        wrangle_shading_network(global_material[0])

    attrib_h = gdp.attribute("geo:prim", "shop_materialpath")
    if attrib_h >= 0:
        shop_materialpaths = gdp.attribProperty(attrib_h, "geo:allstrings")
        for shop in shop_materialpaths:
            wrangle_shading_network(shop)

    # TODO / CONSIDER, for very large number of instance objects it might speed things
    #   up to cache the fact we've already visited a source network.
    #   Store in scenestate?
    #   (This will avoid much of the below on a per instance basis)
    instance_info = get_full_instance_info(obj, now)
    if instance_info is None:
        return
    attrib_h = instance_info.gdp.attribute("geo:point", "shop_materialpath")
    if attrib_h >= 0:
        shop_materialpaths = instance_info.gdp.attribProperty(
            attrib_h, "geo:allstrings"
        )
        for shop in shop_materialpaths:
            wrangle_shading_network(shop)
    return


def output_mediums(obj, wrangler, now):
    """Output the any mediums associated with the Soho Object"""
    exterior = obj.wrangleString(wrangler, "pbrt_exterior", now, [None])[0]
    interior = obj.wrangleString(wrangler, "pbrt_interior", now, [None])[0]

    exterior = wrangle_medium(exterior)
    interior = wrangle_medium(interior)

    return interior, exterior


def output_instances(obj, wrangler, now):
    """Define any instances referenced by the Soho Object

    This method takes an object and based on its parms and point attributes
    will iterate over any found instances and output them so they can be
    later referenced.
    """

    for instance in find_referenced_instances(obj):
        if instance in scene_state.instanced_geo:
            # If we've already emitted this reference geometry
            # then continue so we don't have duplicate definitions
            # this can happen if multiple instance nodes reference
            # the same geo
            continue
        scene_state.instanced_geo.add(instance)

        # Since a referenced geo might not be displayed, output its
        # mediums if any.
        # TODO this works but is a bit magic, rethink this and see if there
        # is a better approach.
        instance_obj = soho.getObject(instance)
        output_materials(instance_obj, wrangler, now)
        output_mediums(instance_obj, wrangler, now)

        with api.ObjectBlock(instance), api.AttributeBlock():
            soho_obj = soho.getObject(instance)
            wrangle_obj(soho_obj, wrangler, now)
        print()
    return


def header():  # pragma: no coverage
    """Output informative header about state"""
    # Disable the header in the event we want to diff files for testing.
    if "SOHO_PBRT_NO_HEADER" in os.environ:
        return
    if scene_state.ver is not None:
        api.Comment("Houdini Version %s" % scene_state.ver)
    api.Comment("Generation Time: %s" % time.strftime("%b %d, %Y at %H:%M:%S"))
    if scene_state.hipfile:
        api.Comment("Hip File: %s" % scene_state.hipfile)
    if scene_state.rop is not None:
        api.Comment("Output Driver: %s" % scene_state.rop)
    if scene_state.now is not None:
        api.Comment("Output Time: %s" % scene_state.now)
    if scene_state.fps:
        api.Comment("Output FPS: %s" % scene_state.fps)
    print()
    return


def footer(start_time):  # pragma: no coverage
    # Disable the header in the event we want to diff files for testing.
    if "SOHO_PBRT_NO_HEADER" in os.environ:
        return
    export_time = time.time() - start_time
    api.Comment("Total export time %0.02f seconds" % export_time)


def output_transform_times(cam, now):
    """Output the TransformTimes for the scene"""
    do_mb = cam.getDefaultedInt("allowmotionblur", now, [0])
    if not do_mb[0]:
        return
    window = cam.getDefaultedFloat("pbrt_motionwindow", now, [None])
    if window[0] is None:
        return
    api.TransformTimes(window[0], window[1])
    print()
    return


def scene_renderables(now):

    # For now we will not be using wranglers
    wrangler = None

    # Output Materials
    api.Comment("=" * 50)
    api.Comment("NamedMaterial Definitions")
    for obj in soho.objectList("objlist:instance"):
        output_materials(obj, wrangler, now, skip_included=True)

    print()

    # Output NamedMediums
    api.Comment("=" * 50)
    api.Comment("NamedMedium Definitions")
    for obj in soho.objectList("objlist:instance"):
        output_mediums(obj, wrangler, now)

    print()

    # Output Object Instances for Fast Instancing
    api.Comment("=" * 50)
    api.Comment("Object Instance Definitions")
    for obj in soho.objectList("objlist:instance"):
        # TODO do we need to set colorspace here?
        #      probably not, but need to create some test cases
        output_instances(obj, wrangler, now)

    print()

    # Output Objects
    api.Comment("=" * 50)
    api.Comment("Object Definitions")
    for obj in soho.objectList("objlist:instance"):
        api.Comment("-" * 50)
        api.Comment(obj.getName())
        with api.AttributeBlock():
            val = []
            if obj.evalString("pbrt_colorspace", now, val):
                api.ColorSpace(val[0])
            wrangle_obj(obj, wrangler, now, concat_xform=True)
        print()


def render(cam, now):
    """Main render entry point"""

    start_time = time.time()

    # For now we will not be using wranglers
    wrangler = None

    header()

    for name, value in wrangle_options(cam, wrangler, now):
        api.Option(name, value)

    val = []
    if cam.evalString("pbrt_colorspace", now, val):
        api.ColorSpace(val[0])

    print()

    api.Film(*wrangle_film(cam, wrangler, now))
    api.Filter(*wrangle_filter(cam, wrangler, now))
    api.Sampler(*wrangle_sampler(cam, wrangler, now))
    api.Integrator(*wrangle_integrator(cam, wrangler, now))
    api.Accelerator(*wrangle_accelerator(cam, wrangler, now))

    print()

    # wrangle_camera will output api.Transforms
    api.Comment(cam.getName())
    api.Camera(*wrangle_camera(cam, wrangler, now))

    print()

    output_transform_times(cam, now)

    # We will stash the global exterior and interior values in case they need
    # to be compared against later.
    interior, exterior = output_mediums(cam, wrangler, now)
    scene_state.exterior = exterior
    scene_state.interior = interior
    if exterior:
        api.MediumInterface("", exterior)
        print()

    api.WorldBegin()

    print()

    # Output Lights
    api.Comment("=" * 50)
    api.Comment("Light Definitions")
    for light in soho.objectList("objlist:light"):
        api.Comment(light.getName())
        with api.AttributeBlock():
            val = []
            if light.evalString("pbrt_colorspace", now, val):
                api.ColorSpace(val[0])
            wrangle_light(light, wrangler, now)
        print()

    print()
    scene_renderables(now)
    print()

    # This was previously an api.WorldEnd() which would have provided the deindent
    soho.indent(-1)
    footer(start_time)

    return


def archive(now):
    """Main archive entry point"""

    start_time = time.time()

    header()

    api.AttributeBegin()
    print()
    scene_renderables(now)
    print()
    api.AttributeEnd()

    footer(start_time)

    return
