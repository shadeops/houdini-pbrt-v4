import os
import shutil
import filecmp
import unittest

import hou

CLEANUP_FILES = False

# Disable headers in pbrt scene files as they have time info
os.environ["SOHO_PBRT_NO_HEADER"] = "1"


def build_checker_material():
    matte = hou.node("/mat").createNode("pbrt_material_diffuse", run_init_scripts=False)
    checks = hou.node("/mat").createNode(
        "pbrt_texture_checkerboard", run_init_scripts=False
    )
    checks.parm("signature").set("s")
    checks.parmTuple("tex1_s").set([0.1, 0.1, 0.1])
    checks.parmTuple("tex2_s").set([0.375, 0.5, 0.5])
    checks.parm("uscale").set(10)
    checks.parm("vscale").set(10)
    matte.setNamedInput("reflectance", checks, "output")
    return matte


def clear_mat():
    for child in hou.node("/mat").children():
        child.destroy()


def build_envlight():
    env = hou.node("/obj").createNode("envlight")
    env.parm("light_intensity").set(0.5)
    return env


def build_spherelight():
    light = hou.node("/obj").createNode("hlight")
    light.parm("light_type").set("sphere")
    light.parmTuple("areasize").set([2, 2])
    light.parmTuple("t").set([0, 10, 0])
    light.parm("light_intensity").set(10)
    return light


def build_cam():
    cam = hou.node("/obj").createNode("cam")
    cam.parmTuple("t").set([8, 8, 8])
    cam.parmTuple("r").set([-35, 48, 0])
    cam.parmTuple("res").set([320, 240])
    return cam


def build_zcam():
    cam = hou.node("/obj").createNode("cam")
    cam.parmTuple("t").set([0, 0, 10])
    cam.parmTuple("res").set([320, 240])
    return cam


def build_geo(name=None):
    geo = hou.node("/obj").createNode("geo", node_name=name)
    for child in geo.children():
        child.destroy()
    return geo


def build_instance():
    instance = hou.node("/obj").createNode("instance")
    for child in instance.children():
        child.destroy()
    return instance


def build_ground():
    ground = hou.node("/obj").createNode("geo")
    for child in ground.children():
        child.destroy()
    ground.createNode("grid")
    return ground


def build_volume(geo, name="", res=8, rgb=False, density_ramp=True):
    volume = geo.createNode("volume")
    volume.parm("name").set(name)
    volume.parm("samplediv").set(res)
    volume.parmTuple("initialval").set([1, 1, 1])

    if rgb:
        volume.parm("rank").set("vector")

    if not density_ramp:
        return volume

    wrangler = geo.createNode("volumewrangle")
    wrangler.setFirstInput(volume)
    wrangle_field = name if name else "density"
    if rgb:
        wrangler.parm("snippet").set(
            "vector res = set(i@resx, i@resy, i@resz);\n"
            "vector i = set(i@ix, i@iy, i@iz);\n"
            "v@{} = fit(i,0,res,0,1);".format(wrangle_field)
        )
    else:
        wrangler.parm("snippet").set(
            "@{} = fit(i@ix,0,i@resx,0,1);".format(wrangle_field)
        )

    return wrangler


def build_vdb(geo, name="density", res=8, density_ramp=True):
    volume = build_volume(geo, name=name, res=res, density_ramp=density_ramp)
    vdb = geo.createNode("convertvdb")
    vdb.parm("conversion").set("vdb")
    vdb.setFirstInput(volume)
    return vdb


def build_rop(filename=None, diskfile=None):
    rop = hou.node("/out").createNode("pbrt")
    ptg = rop.parmTemplateGroup()
    precision = hou.properties.parmTemplate("pbrt-v4", "soho_precision")
    almostzero = hou.properties.parmTemplate("pbrt-v4", "soho_almostzero")
    ptg.append(precision)
    ptg.append(almostzero)
    rop.setParmTemplateGroup(ptg)
    rop.parm("soho_precision").set(2)
    rop.parm("soho_almostzero").set(0.001)
    rop.parm("soho_outputmode").set(1)
    rop.parm("pbrt_geo_location").set("../../geometry")
    if diskfile:
        rop.parm("soho_diskfile").set(diskfile)
    if filename:
        rop.parm("filename").set(filename)
    return rop


def build_archive(diskfile=None):
    rop = hou.node("/out").createNode("pbrtarchive")
    ptg = rop.parmTemplateGroup()
    precision = hou.properties.parmTemplate("pbrt-v4", "soho_precision")
    almostzero = hou.properties.parmTemplate("pbrt-v4", "soho_almostzero")
    ptg.append(precision)
    ptg.append(almostzero)
    rop.setParmTemplateGroup(ptg)
    rop.parm("soho_precision").set(2)
    rop.parm("soho_almostzero").set(0.001)
    if diskfile:
        rop.parm("soho_diskfile").set(diskfile)
    return rop


class NoTest(object):
    pass


class TestParamBase(unittest.TestCase):

    # In order to import the Soho related PBRT modules we need to
    # invoke a render first. While hacky this avoids having to setup
    # custom python path.
    @classmethod
    def setUpClass(cls):
        cls.cam = build_cam()
        cls.rop = build_rop()
        cls.rop.parm("filename").set("/dev/null")

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.rop.render()
        from PBRTnodes import PBRTParam

        self.PBRTParam = PBRTParam

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            self.PBRTParam("cake", "my_name", "foo")

    def test_invalid_equal(self):
        a = self.PBRTParam("float", "my_name", 1)
        with self.assertRaises(TypeError):
            a == "dog"

    def test_invalid_notequal(self):
        a = self.PBRTParam("float", "my_name", 1)
        with self.assertRaises(TypeError):
            a != "dog"

    def test_rgb_is_spectrum(self):
        param = self.PBRTParam("rgb", "my_name", [1, 2, 3])
        self.assertEqual(param.type, "spectrum")

    def test_rgb_string_is_equal(self):
        param = self.PBRTParam("rgb", "my_name", [1, 2, 3])
        self.assertEqual(str(param), "rgb my_name [ 1 2 3 ]")

    def test_rgb_string_is_notequal(self):
        param = self.PBRTParam("rgb", "my_name", [1, 2, 3])
        self.assertNotEqual(str(param), "spectrum my_name [ 0 0 0 ]")

    def test_rgb_is_equal(self):
        a = self.PBRTParam("rgb", "my_name", [1, 2, 3])
        b = self.PBRTParam("color", "my_name", [0, 1, 0])
        self.assertEqual(a, b)

    def test_rgb_is_notequal(self):
        a = self.PBRTParam("rgb", "my_name", [1, 2, 3])
        b = self.PBRTParam("float", "my_name", [0])
        self.assertNotEqual(a, b)

    def test_shorten_str(self):
        param = self.PBRTParam("spectrum", "my_name", [400, 1, 500, 1, 600, 1])
        self.assertEqual(str(param), "spectrum my_name [ 400 1 500 ... ]")

    def test_shorten_generator(self):
        gen = (x for x in [400, 1, 500, 1, 600, 1])
        param = self.PBRTParam("spectrum", "my_name", gen)
        self.assertEqual(str(param), "spectrum my_name [ ... ]")


class TestRoot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    @property
    def testfile(self):
        return "tests/tmp/%s.pbrt" % "/".join(self.id().split(".")[1:])

    @property
    def basefile(self):
        return "tests/scenes/%s.pbrt" % "/".join(self.id().split(".")[1:])

    @property
    def name(self):
        return self.id().split(".")[-1]


class TestROP(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.cam = build_cam()

    def setUp(self):
        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_filter_gaussian(self):
        self.rop.parm("filter").set("gaussian")
        self.rop.parmTuple("filter_radius").set([1.5, 1.5])
        self.rop.parm("gauss_alpha").set(3)
        self.compare_scene()

    def test_filter_mitchell(self):
        self.rop.parm("filter").set("mitchell")
        self.rop.parm("mitchell_B").set(0.3)
        self.rop.parm("mitchell_C").set(0.3)
        self.compare_scene()

    def test_filter_sinc(self):
        self.rop.parm("filter").set("sinc")
        self.rop.parm("sinc_tau").set(4)
        self.compare_scene()

    def test_sampler_stratified(self):
        self.rop.parm("sampler").set("stratified")
        self.compare_scene()

    def test_accelerator_kdtree(self):
        self.rop.parm("accelerator").set("kdtree")
        self.compare_scene()


class TestArchive(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.cam = build_cam()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.geo = build_ground()
        self.rop = build_archive(diskfile=self.testfile)

    def tearDown(self):
        self.geo.destroy()
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_singlegeo(self):
        self.rop.parm("vobject").set(self.geo.path())
        self.compare_scene()


class TestLights(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.cam = build_cam()
        cls.geo = build_ground()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.light = hou.node("/obj").createNode("hlight")
        self.light.parm("ty").set(1.5)
        self.light.parm("rx").set(-90)
        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.light.destroy()
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_pointlight_no_color(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.compare_scene()

    def test_spotlight_no_color(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parm("coneenable").set(True)
        self.compare_scene()

    def test_projectorlight_no_color(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parm("coneenable").set(True)
        self.light.parm("projmap").set("../../resources/tex.exr")
        self.compare_scene()

    def test_goniometriclight_no_color(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parm("areamap").set("../../resources/tex.exr")
        self.compare_scene()

    def test_pointlight(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()

    def test_spotlight(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.light.parm("coneenable").set(True)
        self.compare_scene()

    def test_projectorlight(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.light.parm("coneenable").set(True)
        self.light.parm("projmap").set("../../resources/tex.exr")
        self.compare_scene()

    def test_goniometriclight(self):
        self.light.parm("light_type").set("point")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.light.parm("areamap").set("../../resources/tex.exr")
        self.compare_scene()

    def test_distantlight(self):
        self.light.parm("light_type").set("distant")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()

    def test_spherelight(self):
        self.light.parm("light_type").set("sphere")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()

    def test_spherelight_rotated(self):
        self.light.parm("light_type").set("sphere")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.light.parmTuple("r").set([15, 30, 45])
        self.compare_scene()

    def test_tubelight(self):
        self.light.parm("light_type").set("tube")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()

    def test_disklight(self):
        self.light.parm("light_type").set("disk")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()

    def test_gridlight(self):
        self.light.parm("light_type").set("grid")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()

    def test_gridlight_tex(self):
        self.light.parm("light_type").set("grid")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.light.parm("light_texture").set("../../resources/tex.exr")
        self.compare_scene()

    def test_sunlight(self):
        self.light.parm("light_type").set("sun")
        self.light.parm("light_intensity").set(5)
        self.light.parmTuple("light_color").set([0.5, 0.75, 1])
        self.compare_scene()


class TestGeoLight(TestRoot):
    @classmethod
    def setUpClass(cls):
        cls.cam = build_cam()
        cls.geo = build_ground()

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def setUp(self):
        self.light = hou.node("/obj").createNode("hlight")
        self.light.parm("ty").set(1.5)
        self.light.parm("rx").set(-90)
        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)
        self.box = hou.node("/obj").createNode("geo")
        self.box.createNode("box")
        self.box.setDisplayFlag(False)

    def tearDown(self):
        self.box.destroy()
        self.light.destroy()
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def test_geolight(self):
        self.light.parm("light_type").set("geo")
        self.light.parm("areageometry").set(self.box.path())
        self.light.parm("light_intensity").set(5)
        self.compare_scene()

    def test_geolight_no_geo(self):
        self.light.parm("light_type").set("geo")
        self.light.parm("areageometry").set("")
        self.light.parm("light_intensity").set(5)
        self.compare_scene()


class TestInstance(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.env = build_envlight()
        cls.cam = build_cam()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.geo1 = build_geo()
        self.geo1.createNode("sphere")
        self.geo1.setDisplayFlag(False)
        self.geo2 = build_geo()
        self.geo2.createNode("sphere")
        self.geo2.setDisplayFlag(False)
        self.instance = build_instance()
        self.mat = build_checker_material()

        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

        self.geo1.parm("shop_materialpath").set(self.mat.path())
        self.geo2.parm("shop_materialpath").set(self.mat.path())

    def tearDown(self):
        self.geo1.destroy()
        self.geo2.destroy()
        self.instance.destroy()
        self.rop.destroy()
        clear_mat()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_instance(self):
        add_sop = self.instance.createNode("add")
        add_sop.parm("usept0").set(True)
        self.instance.parm("instancepath").set(self.geo1.path())
        self.compare_scene()

    def test_full_instance(self):
        add_sop = self.instance.createNode("add")
        add_sop.parm("usept0").set(True)
        self.instance.parm("instancepath").set(self.geo1.path())
        self.instance.parm("ptinstance").set("on")
        self.compare_scene()

    def test_fast_instance(self):
        add_sop = self.instance.createNode("add")
        add_sop.parm("usept0").set(True)
        self.instance.parm("instancepath").set(self.geo1.path())
        self.instance.parm("ptinstance").set("fast")
        self.compare_scene()

    def test_full_pt_instance(self):
        add_sop = self.instance.createNode("add")
        add_sop.parm("points").set(2)
        add_sop.parm("usept0").set(True)
        add_sop.parm("usept1").set(True)
        add_sop.parmTuple("pt1").set([2, 0, 0])
        attrib1_sop = self.instance.createNode("attribcreate")
        attrib1_sop.setFirstInput(add_sop)
        attrib1_sop.parm("group").set("0")
        attrib1_sop.parm("name1").set("instance")
        attrib1_sop.parm("type1").set("index")
        attrib1_sop.parm("string1").set(self.geo1.path())
        attrib2_sop = self.instance.createNode("attribcreate")
        attrib2_sop.setFirstInput(attrib1_sop)
        attrib2_sop.parm("group").set("1")
        attrib2_sop.parm("name1").set("instance")
        attrib2_sop.parm("type1").set("index")
        attrib2_sop.parm("string1").set(self.geo2.path())
        attrib2_sop.setRenderFlag(True)
        self.instance.parm("instancepath").set(self.geo1.path())
        self.instance.parm("ptinstance").set("on")
        self.compare_scene()

    def test_fast_pt_instance(self):
        add_sop = self.instance.createNode("add")
        add_sop.parm("points").set(2)
        add_sop.parm("usept0").set(True)
        add_sop.parm("usept1").set(True)
        add_sop.parmTuple("pt1").set([2, 0, 0])
        attrib1_sop = self.instance.createNode("attribcreate")
        attrib1_sop.setFirstInput(add_sop)
        attrib1_sop.parm("group").set("0")
        attrib1_sop.parm("name1").set("instance")
        attrib1_sop.parm("type1").set("index")
        attrib1_sop.parm("string1").set(self.geo1.path())
        attrib2_sop = self.instance.createNode("attribcreate")
        attrib2_sop.setFirstInput(attrib1_sop)
        attrib2_sop.parm("group").set("1")
        attrib2_sop.parm("name1").set("instance")
        attrib2_sop.parm("type1").set("index")
        attrib2_sop.parm("string1").set(self.geo2.path())
        attrib2_sop.setRenderFlag(True)
        self.instance.parm("instancepath").set(self.geo1.path())
        self.instance.parm("ptinstance").set("fast")
        self.compare_scene()


class TestMediums(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.geo = build_geo()
        self.cam = build_cam()
        self.lgt = build_spherelight()
        self.geo.createNode("sphere")
        self.none = hou.node("/mat").createNode("pbrt_material_none")

        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.geo.destroy()
        self.rop.destroy()
        self.cam.destroy()
        self.lgt.destroy()
        clear_mat()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def add_medium_shaders(self, node, interior=None, exterior=None):
        if interior is None and exterior is None:
            return None
        ptg = node.parmTemplateGroup()
        if interior:
            interior_pt = hou.properties.parmTemplate("pbrt-v4", "pbrt_interior")
            ptg.append(interior_pt)
        if exterior:
            exterior_pt = hou.properties.parmTemplate("pbrt-v4", "pbrt_exterior")
            ptg.append(exterior_pt)
        node.setParmTemplateGroup(ptg)
        if interior:
            node.parm("pbrt_interior").set(interior)
        if exterior:
            node.parm("pbrt_exterior").set(exterior)
        return None

    def test_interior_homogeneous(self):
        medium = hou.node("/mat").createNode("pbrt_medium_homogeneous")
        self.add_medium_shaders(self.geo, interior=medium.path())
        self.geo.parm("shop_materialpath").set(self.none.path())
        self.compare_scene()

    def test_interior_cloud(self):
        medium = hou.node("/mat").createNode("pbrt_medium_cloud")
        medium.parmTuple("p0").set([-1, -1, -1])
        medium.parmTuple("p1").set([1, 1, 1])
        medium.parm("density").set(10)
        self.add_medium_shaders(self.geo, interior=medium.path())
        self.geo.parm("shop_materialpath").set(self.none.path())
        self.compare_scene()

    def test_interior_nanovdb(self):
        medium = hou.node("/mat").createNode("pbrt_medium_nanovdb")
        medium.parmTuple("sigma_s").set([0.9, 1.2, 1.5])
        medium.parm("filename").set("../../resources/sphere.nvdb")
        self.add_medium_shaders(self.geo, interior=medium.path())
        self.geo.parm("shop_materialpath").set(self.none.path())
        self.compare_scene()

    def test_exterior_cam(self):
        air = hou.node("/mat").createNode("pbrt_medium_homogeneous")
        air.parmTuple("sigma_a").set([0.01, 0.01, 0.01])
        air.parm("scale").set(0.1)
        self.add_medium_shaders(self.cam, exterior=air.path())
        self.compare_scene()

    def test_exterior_cam_interior_obj(self):
        # TODO / NOTE:
        # I don't believe this test is givign the expected results
        # as there is no mediums attached to the lights.
        # This is a lack of understanding on how pbrt-v4 works and
        # will require some experiments
        air = hou.node("/mat").createNode("pbrt_medium_homogeneous")
        air.parmTuple("sigma_a").set([0.01, 0.01, 0.01])
        air.parm("scale").set(0.1)
        geo_fog = hou.node("/mat").createNode("pbrt_medium_homogeneous")
        geo_fog.parmTuple("sigma_a").set([0.01, 0.01, 0.01])
        geo_fog.parmTuple("sigma_s").set([1, 0.1, 0.05])
        geo_fog.parm("scale").set(10)
        self.add_medium_shaders(self.cam, exterior=air.path())
        self.add_medium_shaders(self.lgt, exterior=air.path())
        self.add_medium_shaders(self.geo, interior=geo_fog.path(), exterior=air.path())
        self.geo.parm("shop_materialpath").set(self.none.path())
        self.compare_scene()


class TestProperties(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.env = build_envlight()
        cls.cam = build_cam()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.geo = build_geo()
        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.geo.destroy()
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_include(self):
        ptg = self.geo.parmTemplateGroup()
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_include")
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_include").set("../../resources/test_include.pbrt")
        self.compare_scene()


class TestMaterials(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.env = build_envlight()
        cls.cam = build_cam()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.geo = build_geo()
        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.geo.destroy()
        self.rop.destroy()
        clear_mat()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_mix_material(self):
        matte1 = hou.node("/mat").createNode("pbrt_material_diffuse")
        matte2 = hou.node("/mat").createNode("pbrt_material_diffuse")
        mix = hou.node("/mat").createNode("pbrt_material_mix")
        mix.setNamedInput("namedmaterial1", matte1, "material")
        mix.setNamedInput("namedmaterial2", matte2, "material")
        self.geo.parm("shop_materialpath").set(mix.path())
        self.compare_scene()

    def test_aliased_parm_material(self):
        coated = hou.node("/mat").createNode("pbrt_material_coatedconductor")
        coated.parm("interface_roughness").set(0.2)
        self.geo.parm("shop_materialpath").set(coated.path())
        self.compare_scene()

    def test_callback_parm_material(self):
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        tex = hou.node("/mat").createNode("pbrt_texture_imagemap")
        tex.parm("filename").set("../../resources/tex.exr")
        tex.parm("signature").set("s")
        tex.parm("auto_gamma").set(False)
        tex.parm("encoding").set("gamma")
        tex.parm("gamma").set(2.1)
        diffuse.setNamedInput("reflectance", tex, "output")
        self.geo.parm("shop_materialpath").set(diffuse.path())
        self.compare_scene()

    def test_displacement_material(self):
        matte = hou.node("/mat").createNode("pbrt_material_diffuse")
        bump = hou.node("/mat").createNode("pbrt_texture_wrinkled")
        matte.setNamedInput("displacement", bump, "output")
        self.geo.parm("shop_materialpath").set(matte.path())
        self.compare_scene()

    def test_checker_material(self):
        space = hou.node("/obj").createNode("null")
        space.parmTuple("t").set([1, 2, 3])
        space.parmTuple("s").set([5, 10, 20])
        matte = hou.node("/mat").createNode("pbrt_material_diffuse")
        checks = hou.node("/mat").createNode("pbrt_texture_checkerboard")
        checks.parm("signature").set("s")
        checks.parm("dimension").set(3)
        checks.parm("texture_space").set(space.path())
        matte.setNamedInput("reflectance", checks, "output")
        self.geo.parm("shop_materialpath").set(matte.path())
        self.compare_scene()

    def test_signature_float_material(self):
        dielectric = hou.node("/mat").createNode("pbrt_material_dieletric")
        dielectric.parm("eta").set(1.3)
        self.geo.parm("shop_materialpath").set(dielectric.path())
        self.compare_scene()

    def test_signature_spectrum_material(self):
        dielectric = hou.node("/mat").createNode("pbrt_material_dieletric")
        dielectric.parm("signature").set("s")
        dielectric.parmTuple("eta_s").set([1.25, 1.5, 1.75])
        self.geo.parm("shop_materialpath").set(dielectric.path())
        self.compare_scene()

    def test_signature_spectrum_texture_material(self):
        dielectric = hou.node("/mat").createNode("pbrt_material_dieletric")
        dielectric.parm("signature").set("s")
        checks = hou.node("/mat").createNode("pbrt_texture_checkerboard")
        checks.parm("signature").set("s")
        dielectric.setNamedInput("eta", checks, "output")
        self.geo.parm("shop_materialpath").set(dielectric.path())
        self.compare_scene()


class TestSpectrum(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.env = build_envlight()
        cls.cam = build_cam()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.geo = build_geo()
        self.material = hou.node("/mat").createNode("pbrt_material_diffuse")
        self.spectrum = hou.node("/mat").createNode("pbrt_spectrum")
        self.material.setNamedInput("reflectance", self.spectrum, "output")

        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

        self.geo.parm("shop_materialpath").set(self.material.path())

    def tearDown(self):
        self.rop.destroy()
        self.geo.destroy()
        self.material.destroy()
        self.spectrum.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_rgb(self):
        self.spectrum.parmTuple("rgb").set([0.25, 0.5, 0.75])
        self.spectrum.parm("type").set("rgb")
        self.compare_scene()

    def test_spd(self):
        self.spectrum.parm("spd").set({"400": "1", "500": "0.5", "600": "0.25"})
        self.spectrum.parm("type").set("spd")
        self.compare_scene()

    def test_file(self):
        self.spectrum.parm("file").set("../../resources/constant.spd")
        self.spectrum.parm("type").set("file")
        self.compare_scene()

    def test_named(self):
        self.spectrum.parm("file").set("metal-Al-k")
        self.spectrum.parm("type").set("file")
        self.compare_scene()

    def test_ramp(self):
        ramp = hou.Ramp([hou.rampBasis.Linear] * 3, (0.0, 0.5, 1.0), (0.25, 1.0, 0.5))
        self.spectrum.parm("ramp").set(ramp)
        self.spectrum.parm("type").set("ramp")
        self.compare_scene()

    def test_blackbody(self):
        self.spectrum.parm("blackbody").set(5000)
        self.spectrum.parm("type").set("blackbody")
        self.compare_scene()


class TestMotionBlur(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.env = build_envlight()

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        self.cam = build_zcam()
        self.geo = build_geo()
        exr = "%s.exr" % self.name
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.geo.destroy()
        self.cam.destroy()
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_obj_mb(self):
        self.geo.parm("tx").setExpression("$FF-1")
        self.rop.parm("allowmotionblur").set(True)
        self.compare_scene()

    def test_cam_mb(self):
        self.cam.parm("tx").setExpression("$FF-1")
        self.rop.parm("allowmotionblur").set(True)
        self.compare_scene()

    def test_motion_window(self):
        self.geo.parm("tx").setExpression("$FF-1")
        self.rop.parm("allowmotionblur").set(True)
        ptg = self.cam.parmTemplateGroup()
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_motionwindow")
        ptg.append(parm)
        self.cam.setParmTemplateGroup(ptg)
        self.cam.parmTuple("pbrt_motionwindow").set([0.25, 0.75])
        self.compare_scene()


class TestShapes(TestRoot):
    @classmethod
    def setUpClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls.cam = build_cam()
        cls.env = build_envlight()
        cls.alpha_tex = hou.node("/mat").createNode("pbrt_texture_dots")
        cls.alpha_tex.parm("uscale").set(10)
        cls.alpha_tex.parm("vscale").set(10)

    @classmethod
    def tearDownClass(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        if CLEANUP_FILES:
            shutil.rmtree("tests/tmp")

    def setUp(self):
        exr = "%s.exr" % self.name
        self.geo = build_geo(self.name)
        self.rop = build_rop(filename=exr, diskfile=self.testfile)

    def tearDown(self):
        self.geo.destroy()
        self.rop.destroy()
        if CLEANUP_FILES:
            os.remove(self.testfile)

    def add_alpha_texture(self):
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_alpha_texture")
        ptg = self.geo.parmTemplateGroup()
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_alpha_texture").set(self.alpha_tex.path())

    def compare_scene(self):
        self.rop.render()
        self.assertTrue(filecmp.cmp(self.testfile, self.basefile))

    def test_sphere(self):
        self.geo.createNode("sphere")
        self.compare_scene()

    def test_sphere_xformed(self):
        sphere = self.geo.createNode("sphere")
        sphere.parmTuple("rad").set([0.5, 0.25, 0.75])
        xform = self.geo.createNode("xform")
        xform.setFirstInput(sphere)
        xform.parmTuple("t").set([0.1, 0.2, 0.3])
        xform.parmTuple("r").set([30, 45, 60])
        xform.setRenderFlag(True)
        self.compare_scene()

    def test_sphere_attribs(self):
        sphere = self.geo.createNode("sphere")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("@zmin=-0.2;\n" "@zmax=0.2;\n" "@phimax=180;\n")
        wrangler.setFirstInput(sphere)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_sphere_many(self):
        sphere = self.geo.createNode("sphere")
        sphere.parm("scale").set(0.1)
        copy = self.geo.createNode("copyxform")
        copy.parm("ncy").set(5)
        copy.parm("tx").set(-0.5)
        copy.setFirstInput(sphere)
        copy.setRenderFlag(True)
        self.compare_scene()

    def test_sphere_alpha(self):
        self.geo.createNode("sphere")
        self.add_alpha_texture()
        self.compare_scene()

    def test_disk(self):
        self.geo.createNode("circle")
        self.compare_scene()

    def test_disk_xformed(self):
        disk = self.geo.createNode("circle")
        disk.parmTuple("rad").set([1.5, 0.75])
        disk.parmTuple("t").set([1, 2, 3])
        disk.parm("ry").set(15)
        disk.parm("scale").set(2)
        self.compare_scene()

    def test_disk_attribs(self):
        disk = self.geo.createNode("circle")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("@innerradius=0.25;\n" "@phimax=180;\n")
        wrangler.setFirstInput(disk)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_disk_many(self):
        disk = self.geo.createNode("circle")
        disk.parm("scale").set(0.1)
        copy = self.geo.createNode("copyxform")
        copy.parm("ncy").set(5)
        copy.parm("tz").set(-0.5)
        copy.setFirstInput(disk)
        copy.setRenderFlag(True)
        self.compare_scene()

    def test_disk_alpha(self):
        self.geo.createNode("circle")
        self.add_alpha_texture()
        self.compare_scene()

    def test_cylinder(self):
        self.geo.createNode("tube")
        self.compare_scene()

    def test_cylinder_caps(self):
        tube = self.geo.createNode("tube")
        tube.parm("cap").set(True)
        self.compare_scene()

    def test_cylinder_xformed(self):
        tube = self.geo.createNode("tube")
        tube.parm("cap").set(True)
        tube.parmTuple("t").set([1, 2, 3])
        tube.parm("rx").set(45)
        tube.parm("radscale").set(1.25)
        self.compare_scene()

    def test_unsupported_cylinder(self):
        tube = self.geo.createNode("tube")
        tube.parm("rad1").set(0.5)
        self.compare_scene()

    def test_cylinder_attribs(self):
        tube = self.geo.createNode("tube")
        tube.parm("cap").set(True)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("@phimax=270;\n")
        wrangler.setFirstInput(tube)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_cylinder_alpha(self):
        tube = self.geo.createNode("tube")
        tube.parm("cap").set(True)
        self.add_alpha_texture()
        self.compare_scene()

    def test_trianglemesh(self):
        self.geo.createNode("box")
        self.compare_scene()

    def test_trianglemesh_open(self):
        line = self.geo.createNode("line")
        line.parm("points").set(4)
        self.compare_scene()

    def test_trianglemesh_polysoup(self):
        box = self.geo.createNode("box")
        soup = self.geo.createNode("polysoup")
        soup.setFirstInput(box)
        soup.setRenderFlag(True)
        self.compare_scene()

    def test_trianglemesh_vtxN(self):
        box = self.geo.createNode("box")
        box.parm("vertexnormals").set(True)
        self.compare_scene()

    def test_trianglemesh_ptN(self):
        box = self.geo.createNode("box")
        normal = self.geo.createNode("normal")
        normal.parm("type").set(0)
        normal.setRenderFlag(True)
        normal.setFirstInput(box)
        self.compare_scene()

    def test_trianglemesh_noauto_ptN(self):
        self.geo.createNode("box")
        ptg = self.geo.parmTemplateGroup()
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_computeN")
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_computeN").set(False)
        self.compare_scene()

    def test_trianglemesh_vtxN_vtxUV(self):
        box = self.geo.createNode("box")
        box.parm("vertexnormals").set(True)
        uvtex = self.geo.createNode("texture")
        uvtex.parm("type").set("polar")
        uvtex.setRenderFlag(True)
        uvtex.setFirstInput(box)
        self.compare_scene()

    def test_trianglemesh_vtxN_ptUV(self):
        box = self.geo.createNode("box")
        box.parm("vertexnormals").set(True)
        uvtex = self.geo.createNode("texture")
        uvtex.parm("type").set("polar")
        uvtex.parm("coord").set("point")
        uvtex.setRenderFlag(True)
        uvtex.setFirstInput(box)
        self.compare_scene()

    def test_trianglemesh_vtxUV_alpha(self):
        box = self.geo.createNode("box")
        uvtex = self.geo.createNode("texture")
        uvtex.parm("type").set("polar")
        uvtex.setRenderFlag(True)
        uvtex.setFirstInput(box)
        self.add_alpha_texture()
        self.compare_scene()

    def test_trianglemesh_ptN_ptS(self):
        box = self.geo.createNode("box")
        frame = self.geo.createNode("polyframe")
        frame.setFirstInput(box)
        frame.parm("tangentu").set("S")
        frame.parm("ortho").set(True)
        frame.parm("style").set("edge1")
        frame.setRenderFlag(True)
        self.compare_scene()

    def test_trianglemesh_faceIndices(self):
        box = self.geo.createNode("box")
        divide = self.geo.createNode("divide")
        divide.setFirstInput(box)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("i@faceIndices = @primnum;")
        wrangler.setFirstInput(divide)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_bilinear_mesh(self):
        box = self.geo.createNode("box")
        box.parm("type").set("mesh")
        box.parm("vertexnormals").set(True)
        uv = self.geo.createNode("texture")
        uv.parm("type").set("rowcol")
        uv.setFirstInput(box)
        uv.setRenderFlag(True)
        self.compare_scene()

    def test_bilinear_mesh_notquad(self):
        box = self.geo.createNode("box")
        box.parm("type").set("mesh")
        box.parm("surftype").set("rows")
        self.compare_scene()

    def test_bilinear_mesh_emissionfilename_prop(self):
        ptg = self.geo.parmTemplateGroup()
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_emissionfilename")
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_emissionfilename").set("../../resources/tex.exr")
        box = self.geo.createNode("box")
        box.parmTuple("divrate").set([2, 2, 2])
        box.parm("type").set("mesh")
        self.compare_scene()

    def test_bilinear_mesh_emissionfilename_attrib(self):
        box = self.geo.createNode("box")
        box.parmTuple("divrate").set([2, 2, 2])
        box.parm("type").set("mesh")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set('s@emissionfilename = "../../resources/tex.exr";')
        wrangler.setFirstInput(box)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_loopsubdiv(self):
        self.geo.createNode("box")
        ptg = self.geo.parmTemplateGroup()
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_rendersubd")
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_rendersubd").set(True)
        self.compare_scene()

    def test_loopsubdiv_levels(self):
        self.geo.createNode("box")
        ptg = self.geo.parmTemplateGroup()
        subd_parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_rendersubd")
        level_parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_subdlevels")
        ptg.append(subd_parm)
        ptg.append(level_parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_rendersubd").set(True)
        self.geo.parm("pbrt_subdlevels").set(2)
        self.compare_scene()

    def test_loopsubdiv_alpha(self):
        self.geo.createNode("box")
        ptg = self.geo.parmTemplateGroup()
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_rendersubd")
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_rendersubd").set(True)
        self.add_alpha_texture()
        self.compare_scene()

    def test_tesselated_metaball(self):
        self.geo.createNode("metaball")
        self.compare_scene()

    def test_tesselated_nurbs(self):
        box = self.geo.createNode("box")
        box.parm("type").set("nurbs")
        self.compare_scene()

    def test_tesselated_bezier(self):
        box = self.geo.createNode("box")
        box.parm("type").set("bezier")
        self.compare_scene()

    def test_unsupported_packed(self):
        box = self.geo.createNode("box")
        pack = self.geo.createNode("pack")
        pack.setFirstInput(box)
        pack.setRenderFlag(True)
        self.compare_scene()

    def test_unsupported_heightfield(self):
        hf = self.geo.createNode("heightfield")
        hf.parmTuple("size").set([10, 10])
        self.compare_scene()

    def test_curve_bezier(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        self.compare_scene()

    def test_curve_bezier_many(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        copy = self.geo.createNode("copyxform")
        copy.parm("ncy").set(5)
        copy.parm("tz").set(-0.5)
        copy.setFirstInput(curve)
        copy.setRenderFlag(True)
        self.compare_scene()

    def test_curve_bezier_long(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set(
            "-1,0,0 -1,0,-1 0,0,-1 "
            "0,0,0 0,0,1 1,0,1 "
            "1,0,0 1,0,-1 1,0,-2 "
            "0,0,-2 -1,0,-2 -2,0,-2 "
            "-2,0,-1 -2,0,0 -2,0,1 "
            "-1,0,1"
        )
        curve.parm("type").set("bezier")
        self.compare_scene()

    def test_curve_bezier_types(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        copy = self.geo.createNode("copyxform")
        copy.parm("ncy").set(3)
        copy.parm("tz").set(-1)
        copy.setFirstInput(curve)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set(
            'if (@primnum%3 == 0) s@curvetype = "ribbon";\n'
            'if (@primnum%3 == 1) s@curvetype = "cylinder";\n'
            'if (@primnum%3 == 2) s@curvetype = "flat";'
        )
        wrangler.setFirstInput(copy)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_curve_bezier_width(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("@width = 0.01;")
        wrangler.setFirstInput(curve)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_curve_bezier_width01(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("@width0 = 0.01;\n" "@width1 = 0.1;")
        wrangler.setFirstInput(curve)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_curve_bezier_vtxwidth(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("vertex")
        wrangler.parm("snippet").set("@width = fit(@ptnum, 0, @numpt-1, 0.01, 0.1);")
        wrangler.setFirstInput(curve)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_curve_bezier_ptwidth(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("point")
        wrangler.parm("snippet").set("@width = fit(@ptnum, 0, @numpt-1, 0.01, 0.1);")
        wrangler.setFirstInput(curve)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_curve_bezier_type_prop(self):
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_curvetype")
        ptg = self.geo.parmTemplateGroup()
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_curvetype").set("cylinder")
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        self.compare_scene()

    def test_curve_bezier_splitdepth_prop(self):
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_splitdepth")
        ptg = self.geo.parmTemplateGroup()
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_splitdepth").set(9)
        curve = self.geo.createNode("curve")
        curve.parm("coords").set(
            "-1,0,0 -1,0,-1 0,0,-1 "
            "0,0,0 0,0,1 1,0,1 "
            "1,0,0 1,0,-1 1,0,-2 "
            "0,0,-2 -1,0,-2 -2,0,-2 "
            "-2,0,-1 -2,0,0 -2,0,1 "
            "-1,0,1"
        )
        curve.parm("type").set("bezier")
        self.compare_scene()

    def test_curve_bezier_ribbon_N(self):
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_curvetype")
        ptg = self.geo.parmTemplateGroup()
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_curvetype").set("ribbon")
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("bezier")
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("point")
        wrangler.parm("snippet").set("v@N = {0,1,0};")
        wrangler.setFirstInput(curve)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    @unittest.skipIf(
        hou.applicationVersion() < (17, 5), "Only supported in Houdini 17.5 and higher"
    )
    def test_curve_bspline(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set("-1,0,0 -1,0,-1 0,0,-1 0,0,0")
        curve.parm("type").set("nurbs")
        self.compare_scene()

    def test_curve_bezier_bad_order(self):
        curve = self.geo.createNode("curve")
        curve.parm("coords").set(
            "-1,0,0 -1,0,-1 0,0,-1 "
            "0,0,0 0,0,1 1,0,1 "
            "1,0,0 1,0,-1 1,0,-2 "
            "0,0,-2 -1,0,-2 -2,0,-2 "
            "-2,0,-1 -2,0,0 -2,0,1 "
            "-1,0,1"
        )
        curve.parm("type").set("bezier")
        curve.parm("order").set(6)
        self.compare_scene()

    def test_curve_bezier_closed(self):
        curve = self.geo.createNode("circle")
        curve.parm("type").set("bezier")
        curve.parm("divs").set(4)
        self.compare_scene()

    def test_volume(self):
        volume = build_volume(self.geo)
        volume.setRenderFlag(True)
        self.compare_scene()

    def test_volume_Lescale(self):
        density = build_volume(self.geo, name="density")
        Lescale = build_volume(self.geo, name="Lescale")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, Lescale)
        merge.setRenderFlag(True)
        self.compare_scene()

    def test_volume_many(self):
        volume = build_volume(self.geo)
        copies = self.geo.createNode("copyxform")
        copies.parm("ncy").set(3)
        copies.parm("tx").set(1)
        copies.setFirstInput(volume)
        copies.setRenderFlag(True)
        self.compare_scene()

    def test_volume_Lescale_many(self):
        density = build_volume(self.geo, name="density")
        Lescale = build_volume(self.geo, name="Lescale")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, Lescale)
        copies = self.geo.createNode("copyxform")
        copies.parm("ncy").set(3)
        copies.parm("tx").set(1)
        copies.setFirstInput(merge)
        copies.setRenderFlag(True)
        self.compare_scene()

    def test_volume_Lescale_many_mapped(self):
        density = build_volume(self.geo, name="density")
        Lescale = build_volume(self.geo, name="Lescale")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, Lescale)
        copies = self.geo.createNode("copyxform")
        copies.parm("ncy").set(3)
        copies.parm("tx").set(1)
        copies.setFirstInput(merge)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("i@medium_grids = @primnum/2;")
        wrangler.setFirstInput(copies)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_volume_Lescale_wrong_res(self):
        density = build_volume(self.geo, name="density")
        Lescale = build_volume(self.geo, res=10, name="Lescale")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, Lescale)
        merge.setRenderFlag(True)
        self.compare_scene()

    def test_volume_rgb(self):
        volume = build_volume(self.geo, rgb=True)
        volume.setRenderFlag(True)
        self.compare_scene()

    def test_volume_rgb_Lescale(self):
        density = build_volume(self.geo, rgb=True, name="density")
        Lescale = build_volume(self.geo, name="Lescale")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, Lescale)
        merge.setRenderFlag(True)
        self.compare_scene()

    def test_volume_floats_rgb(self):
        density_float1 = build_volume(self.geo, name="density")
        density_float2 = build_volume(self.geo, name="density")
        density_rgb = build_volume(self.geo, rgb=True, name="density")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density_float1)
        merge.setInput(1, density_float2)
        merge.setInput(2, density_rgb)
        merge.setRenderFlag(True)
        self.compare_scene()

    def test_volume_attribs(self):
        volume = build_volume(self.geo)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set(
            "@g = 0.2;\n"
            "@scale = 0.5;\n"
            "v@sigma_a = {0.01, 0.02, 0.03};\n"
            "v@sigma_s = {1, 2, 3};\n"
            "v@Le = {2,2,2};\n"
        )
        wrangler.setFirstInput(volume)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_volume_attribs_preset(self):
        volume = build_volume(self.geo)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set(
            's@preset = "Regular Milk";'
            "v@sigma_a = {0.01, 0.02, 0.03};\n"
            "v@sigma_s = {1, 2, 3};\n"
        )
        wrangler.setFirstInput(volume)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_volume_attribs_medium(self):
        medium = hou.node("/mat").createNode("pbrt_medium_uniformgrid")
        medium.parm("g").set(-0.5)
        volume = build_volume(self.geo)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set(
            "v@sigma_a = {{0.01, 0.02, 0.03}};\n"
            "v@sigma_s = {{1, 2, 3}};\n"
            's@pbrt_interior = "{}";'.format(medium.path())
        )
        wrangler.setFirstInput(volume)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_vdb(self):
        vdb = build_vdb(self.geo)
        vdb.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_temperature(self):
        density = build_vdb(self.geo, name="density")
        temperature = build_vdb(self.geo, name="temperature")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, temperature)
        merge.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_many(self):
        vdb = build_vdb(self.geo)
        copies = self.geo.createNode("copyxform")
        copies.parm("ncy").set(3)
        copies.parm("tx").set(1)
        copies.setFirstInput(vdb)
        copies.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_temperature_many(self):
        density = build_vdb(self.geo, name="density")
        temperature = build_vdb(self.geo, name="temperature")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, temperature)
        copies = self.geo.createNode("copyxform")
        copies.parm("ncy").set(3)
        copies.parm("tx").set(1)
        copies.setFirstInput(merge)
        copies.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_temperature_many_mapped(self):
        density = build_vdb(self.geo, name="density")
        temperature = build_vdb(self.geo, name="temperature")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, temperature)
        copies = self.geo.createNode("copyxform")
        copies.parm("ncy").set(3)
        copies.parm("tx").set(1)
        copies.setFirstInput(merge)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set("i@medium_grids = @primnum/2;")
        wrangler.setFirstInput(copies)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_one_density_many_temperature(self):
        density = build_vdb(self.geo, name="density")
        temp1 = build_vdb(self.geo, name="temperature")
        temp2 = build_vdb(self.geo, name="temperature")
        merge = self.geo.createNode("merge")
        merge.setInput(0, density)
        merge.setInput(1, temp1)
        merge.setInput(2, temp2)
        merge.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_attribs_preset(self):
        vdb = build_vdb(self.geo)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set(
            's@preset = "Regular Milk";'
            "v@sigma_a = {0.01, 0.02, 0.03};\n"
            "v@sigma_s = {1, 2, 3};\n"
        )
        wrangler.setFirstInput(vdb)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_vdb_attribs_medium(self):
        medium = hou.node("/mat").createNode("pbrt_medium_nanovdb")
        medium.parm("g").set(-0.5)
        vdb = build_vdb(self.geo)
        wrangler = self.geo.createNode("attribwrangle")
        wrangler.parm("class").set("primitive")
        wrangler.parm("snippet").set(
            "v@sigma_a = {{0.01, 0.02, 0.03}};\n"
            "v@sigma_s = {{1, 2, 3}};\n"
            's@pbrt_interior = "{}";'.format(medium.path())
        )
        wrangler.setFirstInput(vdb)
        wrangler.setRenderFlag(True)
        self.compare_scene()

    def test_geo_materials(self):
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        diffuse.parmTuple("reflectance").set([0.0625, 0.0625, 0.75])
        box = self.geo.createNode("box")
        material = self.geo.createNode("material")
        material.parm("shop_materialpath1").set(diffuse.path())
        material.setFirstInput(box)
        material.setRenderFlag(True)
        self.compare_scene()

    def test_geo_material_overrides(self):
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        box = self.geo.createNode("box")
        material = self.geo.createNode("material")
        material.parm("shop_materialpath1").set(diffuse.path())
        material.parm("num_local1").set(1)
        material.parm("local1_name1").set("reflectance")
        material.parm("local1_type1").set("color")
        material.parmTuple("local1_cval1").set([0.9, 0, 0])
        material.setFirstInput(box)
        material.setRenderFlag(True)
        self.compare_scene()

    def test_geo_material_overrides_spectrum_xyz(self):
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        box = self.geo.createNode("box")
        material = self.geo.createNode("material")
        material.parm("shop_materialpath1").set(diffuse.path())
        material.parm("num_local1").set(1)
        material.parm("local1_name1").set("reflectance:xyz")
        material.parm("local1_type1").set("vector3")
        material.parmTuple("local1_vval1").set([0.9, 0.5, 0.1, 0.0])
        material.setFirstInput(box)
        material.setRenderFlag(True)
        with self.assertRaises(hou.OperationFailed):
            self.rop.render()

    def test_geo_material_overrides_spectrum_file(self):
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        box = self.geo.createNode("box")
        material = self.geo.createNode("material")
        material.parm("shop_materialpath1").set(diffuse.path())
        material.parm("num_local1").set(1)
        material.parm("local1_name1").set("reflectance:spectrum")
        material.parm("local1_type1").set("string")
        material.parmTuple("local1_sval1").set(["../../resources/constant.spd"])
        material.setFirstInput(box)
        material.setRenderFlag(True)
        self.compare_scene()

    def test_geo_material_overrides_spectrum_spd(self):
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        box = self.geo.createNode("box")
        material = self.geo.createNode("material")
        material.parm("shop_materialpath1").set(diffuse.path())
        material.parm("num_local1").set(1)
        material.parm("local1_name1").set("reflectance:spectrum")
        material.parm("local1_type1").set("string")
        material.parmTuple("local1_sval1").set(["[400, 0.5, 500, 1, 600, 0.5]"])
        material.setFirstInput(box)
        material.setRenderFlag(True)
        self.compare_scene()

    def test_geo_ignore_materials(self):
        parm = hou.properties.parmTemplate("pbrt-v4", "pbrt_ignorematerials")
        ptg = self.geo.parmTemplateGroup()
        ptg.append(parm)
        self.geo.setParmTemplateGroup(ptg)
        self.geo.parm("pbrt_ignorematerials").set(True)
        diffuse = hou.node("/mat").createNode("pbrt_material_diffuse")
        diffuse.parmTuple("reflectance").set([0.0625, 0.0625, 0.75])
        box = self.geo.createNode("box")
        material = self.geo.createNode("material")
        material.parm("shop_materialpath1").set(diffuse.path())
        material.setFirstInput(box)
        material.setRenderFlag(True)
        self.compare_scene()


if __name__ == "__main__":
    unittest.main()
