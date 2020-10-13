from __future__ import print_function, division, absolute_import

import hou

import PBRTapi as api

from PBRTstate import scene_state
from PBRTnodes import BaseNode


def wrangle_shading_network(
    node_path,
    name_prefix="",
    name_suffix="",
    use_named=True,
    saved_nodes=None,
    overrides=None,
    root=True,
):

    if node_path in scene_state.invalid_shading_nodes:
        return

    # Depth first, as textures/materials need to be
    # defined before they are referenced

    # Use this to track if a node has been output or not.
    # if the saved_nodes is None, we use the global scene_state
    # otherwise we use the one passed in. This is useful for outputing
    # named materials within a nested Attribute Block.
    if saved_nodes is None:
        saved_nodes = scene_state.shading_nodes

    # NOTE: We prefix and suffix names here so that there are not collisions when
    #       using full point instancing. There is some possible redundancy as the same
    #       network maybe recreated multiple times under different names if the
    #       overrides are the same. A possible optimization for export and PBRT is to
    #       do a prepass and build the networks before and keep a map to the pre-built
    #       networks. For now we'll brute force it.
    presufed_node_path = name_prefix + node_path + name_suffix
    if presufed_node_path in saved_nodes:
        return

    hnode = hou.node(node_path)

    # Material or Texture?
    node = BaseNode.from_node(hnode)
    if node is None:
        api.Comment("Skipping %s since its not a Material or Texture node" % node_path)
        scene_state.invalid_shading_nodes.add(node_path)
        return
    else:
        saved_nodes.add(presufed_node_path)

    node.path_suffix = name_suffix
    node.path_prefix = name_prefix

    if node.directive == "material":
        api_call = api.MakeNamedMaterial if use_named else api.Material
    elif node.directive == "texture":
        api_call = api.Texture
    else:
        return

    paramset = node.paramset_with_overrides(overrides)

    for node_input in node.inputs():
        wrangle_shading_network(
            node_input,
            name_prefix=name_prefix,
            name_suffix=name_suffix,
            use_named=use_named,
            saved_nodes=saved_nodes,
            overrides=overrides,
            root=False,
        )

    colorspace = node.colorspace
    if colorspace is not None:
        api.AttributeBegin()
        api.ColorSpace(colorspace)

    coord_sys = node.coord_sys
    if coord_sys:
        api.TransformBegin()
        api.Transform(coord_sys)

    if api_call == api.Material:
        api_call(node.directive_type, paramset)
    else:
        api_call(node.full_name, node.output_type, node.directive_type, paramset)

    if coord_sys:
        api.TransformEnd()

    if colorspace is not None:
        api.AttributeEnd()

    if api_call == api.MakeNamedMaterial:
        print()

    return
