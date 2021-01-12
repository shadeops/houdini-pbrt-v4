import os
import importlib

import hou
import soho
from soho import SohoParm

# Houdini does not reload modules at each render, so when doing dev work it
# can be a pain to have to manually load modules, by setting this environment
# variable all the modules will be reloaded.
if "SOHO_PBRT_DEV" in os.environ:  # noqa # pragma: no coverage
    import PBRTapi

    importlib.reload(PBRTapi)
    import PBRTstate

    importlib.reload(PBRTstate)
    import PBRTnodes

    importlib.reload(PBRTnodes)
    import PBRTsoho

    importlib.reload(PBRTsoho)
    import PBRTinstancing

    importlib.reload(PBRTinstancing)
    import PBRTshading

    importlib.reload(PBRTshading)
    import PBRTgeo

    importlib.reload(PBRTgeo)
    import PBRTwranglers

    importlib.reload(PBRTwranglers)
    import PBRTscene

    importlib.reload(PBRTscene)

import PBRTscene
from PBRTstate import scene_state


def soho_render():
    control_parms = {
        # The time at which the scene is being rendered
        "now": SohoParm("state:time", "real", [0], False, key="now"),
        "camera": SohoParm("camera", "string", ["/obj/cam1"], False),
    }

    parms = soho.evaluate(control_parms)

    now = parms["now"].Value[0]
    camera = parms["camera"].Value[0]

    options = {}
    if not soho.initialize(now, camera, options):
        soho.error("Unable to initialize rendering module with given camera")

    object_selection = {
        # Candidate object selection
        "vobject": SohoParm("vobject", "string", ["*"], False),
        "alights": SohoParm("alights", "string", ["*"], False),
        "forceobject": SohoParm("forceobject", "string", [""], False),
        "forcelights": SohoParm("forcelights", "string", [""], False),
        "excludeobject": SohoParm("excludeobject", "string", [""], False),
        "excludelights": SohoParm("excludelights", "string", [""], False),
        "sololight": SohoParm("sololight", "string", [""], False),
    }

    for cam in soho.objectList("objlist:camera"):
        break
    else:
        soho.error("Unable to find viewing camera for render")

    objparms = cam.evaluate(object_selection, now)

    stdobject = objparms["vobject"].Value[0]
    stdlights = objparms["alights"].Value[0]
    forceobject = objparms["forceobject"].Value[0]
    forcelights = objparms["forcelights"].Value[0]
    excludeobject = objparms["excludeobject"].Value[0]
    excludelights = objparms["excludelights"].Value[0]
    sololight = objparms["sololight"].Value[0]
    forcelightsparm = "forcelights"

    if sololight:
        stdlights = excludelights = ""
        forcelights = sololight
        forcelightsparm = "sololight"

    # First, we add objects based on their display flags or dimmer values
    soho.addObjects(
        now,
        stdobject,
        stdlights,
        "",
        True,
        geo_parm="vobject",
        light_parm="alights",
        fog_parm="",
    )
    soho.addObjects(
        now,
        forceobject,
        forcelights,
        "",
        False,
        geo_parm="forceobject",
        light_parm=forcelightsparm,
        fog_parm="",
    )
    soho.removeObjects(
        now,
        excludeobject,
        excludelights,
        "",
        geo_parm="excludeobject",
        light_parm="excludelights",
        fog_parm="",
    )

    # Lock off the objects we've selected
    soho.lockObjects(now)

    with hou.undos.disabler(), scene_state:
        if "SOHO_PBRT_DEV" in os.environ:  # pragma: no coverage
            import cProfile

            pr = cProfile.Profile()
            pr.enable()
            try:
                PBRTscene.render(cam, now)
            finally:
                pr.disable()
                pr.dump_stats("hou-prbt.stats")
        else:
            PBRTscene.render(cam, now)
    return


if __name__ in ("__main__", "builtins"):
    soho_render()
