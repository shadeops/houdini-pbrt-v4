from __future__ import print_function, division, absolute_import

import hou

import PBRTapi as api

from PBRTstate import scene_state
from PBRTnodes import BaseNode, ParamSet


def wrangle_shading_network(
    node_path,
    name_prefix="",
    name_suffix="",
    use_named=True,
    exported_nodes=None,
    overrides=None,
    node_cache=None,
    param_cache=None,
):

    if node_path in scene_state.invalid_shading_nodes:
        return

    # Depth first, as textures/materials need to be
    # defined before they are referenced

    # Use this to track if a node has been output or not.
    # if the exported_nodes is None, we use the global scene_state
    # otherwise we use the one passed in. This is useful for outputing
    # named materials within a nested Attribute Block.
    if exported_nodes is None:
        exported_nodes = scene_state.shading_nodes

    # NOTE: We prefix and suffix names here so that there are not collisions when
    #       using full point instancing. There is some possible redundancy as the same
    #       network maybe recreated multiple times under different names if the
    #       overrides are the same. A possible optimization for export and PBRT is to
    #       do a prepass and build the networks before and keep a map to the pre-built
    #       networks. For now we'll brute force it.
    presufed_node_path = name_prefix + node_path + name_suffix
    if presufed_node_path in exported_nodes:
        return

    if isinstance(node_cache, dict):
        if node_path not in node_cache:
            hnode = hou.node(node_path)
            node_cache[node_path] = BaseNode.from_node(hnode)
        node = node_cache[node_path]
    else:
        hnode = hou.node(node_path)
        node = BaseNode.from_node(hnode)

    if node is None:
        api.Comment("Skipping %s since its not a Material or Texture node" % node_path)
        scene_state.invalid_shading_nodes.add(node_path)
        return
    else:
        exported_nodes.add(presufed_node_path)

    node.path_suffix = name_suffix
    node.path_prefix = name_prefix

    # Material or Texture?
    if node.directive == "material":
        api_call = api.MakeNamedMaterial if use_named else api.Material
    elif node.directive == "texture":
        api_call = api.Texture
    else:
        return

    if isinstance(param_cache, dict):
        if node_path not in param_cache:
            param_cache[node_path] = node.paramset
        paramset = ParamSet(param_cache[node_path])
        paramset.update(node.override_paramset(overrides))
    else:
        paramset = node.paramset_with_overrides(overrides)

    for node_input in node.inputs():
        wrangle_shading_network(
            node_input,
            name_prefix=name_prefix,
            name_suffix=name_suffix,
            use_named=use_named,
            exported_nodes=exported_nodes,
            overrides=overrides,
            param_cache=param_cache,
        )

    colorspace = node.colorspace
    if colorspace is not None:
        api.AttributeBegin()
        api.ColorSpace(colorspace)

    coord_sys = node.coord_sys
    if coord_sys:
        api.AttributeBegin()
        api.Transform(coord_sys)

    if api_call == api.Material:
        api_call(node.directive_type, paramset)
    else:
        api_call(node.full_name, node.output_type, node.directive_type, paramset)

    if coord_sys:
        api.AttributeEnd()

    if colorspace is not None:
        api.AttributeEnd()

    if api_call == api.MakeNamedMaterial:
        print()

    return
