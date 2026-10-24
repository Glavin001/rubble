"""RubbleDemo's materials, built headless in the editor (scripts/unreal/build-content.sh).

Generated into /Game/Rubble/Materials (never committed; this script is the source):

  M_RubbleSurface   PBR surface: base colour from per-instance custom data 0..2 when an instanced
                    mesh supplies it (else the BaseColor parameter), world-space grit, roughness and
                    metal parameters. MI_Concrete, MI_Brick, MI_Timber, MI_Steel, MI_Glass,
                    MI_Rubber, MI_Plastic and MI_Interior (fracture faces) are its instances.
  M_RubbleCarPaint  glossy metallic-flake paint (UE 5.8's Python exposes no clear-coat pins, so it
                    is DefaultLit); vertex colour R is the soft body's strain (G = 0 marks a
                    skinned mesh), which turns paint into scratched, dented bare metal.
  M_RubbleGround    procedural asphalt / concrete (MI_GroundYard, MI_GroundHall, MI_GroundLot).
  M_RubbleDust      translucent dust puffs, opacity from custom data 3.
  M_RubbleSpark     additive sparks, glow from custom data 3.

Every material is flagged for instanced static meshes, which the renderers draw with.
"""

import unreal

ROOT = "/Game/Rubble/Materials"
tools = unreal.AssetToolsHelpers.get_asset_tools()
mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary
MP = unreal.MaterialProperty


def log(message: str):
    unreal.log("build_content: " + message)


def fresh(name: str, cls, factory):
    path = f"{ROOT}/{name}"
    if eal.does_asset_exist(path):
        eal.delete_asset(path)
    return tools.create_asset(name, ROOT, cls, factory)


def material(name: str):
    m = fresh(name, unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("used_with_instanced_static_meshes", True)
    return m


def expr(m, cls, x, y, **props):
    e = mel.create_material_expression(m, cls, x, y)
    for key, value in props.items():
        e.set_editor_property(key, value)
    return e


def scalar(m, name, value, x, y):
    return expr(m, unreal.MaterialExpressionScalarParameter, x, y, parameter_name=name, default_value=value)


def vector(m, name, color, x, y):
    return expr(m, unreal.MaterialExpressionVectorParameter, x, y, parameter_name=name, default_value=unreal.LinearColor(*color))


def const(m, value, x, y):
    return expr(m, unreal.MaterialExpressionConstant, x, y, r=value)


def link(a, a_out, b, b_in):
    mel.connect_material_expressions(a, a_out, b, b_in)


def binary(m, cls, a, b, x, y, a_out="", b_out=""):
    e = expr(m, cls, x, y)
    link(a, a_out, e, "A")
    link(b, b_out, e, "B")
    return e


def lerp(m, a, b, alpha, x, y):
    e = expr(m, unreal.MaterialExpressionLinearInterpolate, x, y)
    link(a, "", e, "A")
    link(b, "", e, "B")
    link(alpha, "", e, "Alpha")
    return e


def noise(m, scale, x, y, lo=0.0, hi=1.0, levels=4):
    position = expr(m, unreal.MaterialExpressionWorldPosition, x - 250, y)
    n = expr(m, unreal.MaterialExpressionNoise, x, y, scale=scale, levels=levels, output_min=lo, output_max=hi, quality=1)
    link(position, "", n, "Position")
    return n


def instance(name: str, parent, scalars=None, vectors=None):
    mi = fresh(name, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(mi, parent)
    for key, value in (scalars or {}).items():
        mel.set_material_instance_scalar_parameter_value(mi, key, value)
    for key, value in (vectors or {}).items():
        mel.set_material_instance_vector_parameter_value(mi, key, unreal.LinearColor(*value))
    return mi


def surface():
    m = material("M_RubbleSurface")
    # Tint: custom data 0..2 from an instanced mesh; -1 (the default, a plain mesh) means "use BaseColor".
    channels = [expr(m, unreal.MaterialExpressionPerInstanceCustomData, -1400, -500 + 120 * i, data_index=i, const_default_value=-1.0) for i in range(3)]
    rg = binary(m, unreal.MaterialExpressionAppendVector, channels[0], channels[1], -1200, -450)
    tint = binary(m, unreal.MaterialExpressionAppendVector, rg, channels[2], -1050, -420)
    base = vector(m, "BaseColor", (0.5, 0.49, 0.46, 1.0), -1050, -250)
    pick = expr(m, unreal.MaterialExpressionIf, -850, -400)
    link(channels[0], "", pick, "A")
    link(const(m, 0.0, -1050, -150), "", pick, "B")
    link(tint, "", pick, "A > B")
    link(tint, "", pick, "A == B")
    link(base, "", pick, "A < B")
    # Grit: two octaves of world-space noise darken and roughen the surface a little.
    grit = scalar(m, "Grit", 0.25, -1050, 0)
    fine = noise(m, 0.08, -1050, 150)
    coarse = noise(m, 0.006, -1050, 350)
    mix = binary(m, unreal.MaterialExpressionMultiply, fine, coarse, -800, 250)
    one = const(m, 1.0, -800, 50)
    shade = lerp(m, one, mix, grit, -650, 100)
    color = binary(m, unreal.MaterialExpressionMultiply, pick, shade, -450, -300)
    mel.connect_material_property(color, "", MP.MP_BASE_COLOR)
    rough = scalar(m, "Roughness", 0.8, -650, 300)
    rough_var = binary(m, unreal.MaterialExpressionMultiply, fine, grit, -650, 420)
    rough_sum = binary(m, unreal.MaterialExpressionAdd, rough, rough_var, -450, 350)
    saturate = expr(m, unreal.MaterialExpressionSaturate, -300, 350)
    link(rough_sum, "", saturate, "")
    mel.connect_material_property(saturate, "", MP.MP_ROUGHNESS)
    mel.connect_material_property(scalar(m, "Metallic", 0.0, -450, 500), "", MP.MP_METALLIC)
    mel.connect_material_property(scalar(m, "Specular", 0.5, -450, 600), "", MP.MP_SPECULAR)
    mel.recompile_material(m)
    return m


def car_paint():
    m = material("M_RubbleCarPaint")
    paint = vector(m, "Paint", (0.05, 0.18, 0.55, 1.0), -1100, -400)
    bare = vector(m, "Bare", (0.42, 0.42, 0.44, 1.0), -1100, -250)
    vc = expr(m, unreal.MaterialExpressionVertexColor, -1300, 0)
    # Damage = strain (R) on skinned meshes only (G = 0); plain meshes have white vertex colour.
    one = const(m, 1.0, -1300, 200)
    not_g = binary(m, unreal.MaterialExpressionSubtract, one, vc, -1100, 150, b_out="G")
    strain = binary(m, unreal.MaterialExpressionMultiply, vc, not_g, -950, 50, a_out="R")
    scratches = noise(m, 0.35, -1100, 350, lo=0.4, hi=1.6, levels=3)
    raw = binary(m, unreal.MaterialExpressionMultiply, strain, scratches, -800, 150)
    damage = expr(m, unreal.MaterialExpressionSaturate, -650, 150)
    link(raw, "", damage, "")
    mel.connect_material_property(lerp(m, paint, bare, damage, -450, -300), "", MP.MP_BASE_COLOR)
    # Paint: some metal flake, glossy; bare metal: fully metallic, rougher, scratched.
    flake = noise(m, 2.5, -900, 450, lo=0.85, hi=1.0, levels=1)
    paint_metal = binary(m, unreal.MaterialExpressionMultiply, const(m, 0.35, -650, 300), flake, -550, 300)
    mel.connect_material_property(lerp(m, paint_metal, const(m, 1.0, -650, 380), damage, -400, 320), "", MP.MP_METALLIC)
    mel.connect_material_property(lerp(m, const(m, 0.16, -650, 460), const(m, 0.55, -650, 540), damage, -450, 480), "", MP.MP_ROUGHNESS)
    mel.connect_material_property(const(m, 0.7, -450, 620), "", MP.MP_SPECULAR)
    mel.recompile_material(m)
    return m


def ground():
    m = material("M_RubbleGround")
    base = vector(m, "BaseColor", (0.07, 0.07, 0.075, 1.0), -900, -300)
    speck = noise(m, 0.4, -900, -100, lo=0.75, hi=1.25, levels=2)
    patches = noise(m, 0.002, -900, 150, lo=0.88, hi=1.08, levels=4)
    shade = binary(m, unreal.MaterialExpressionMultiply, speck, patches, -650, 0)
    mel.connect_material_property(binary(m, unreal.MaterialExpressionMultiply, base, shade, -400, -200), "", MP.MP_BASE_COLOR)
    rough = scalar(m, "Roughness", 0.85, -650, 250)
    wet = binary(m, unreal.MaterialExpressionMultiply, rough, patches, -400, 250)
    sat = expr(m, unreal.MaterialExpressionSaturate, -250, 250)
    link(wet, "", sat, "")
    mel.connect_material_property(sat, "", MP.MP_ROUGHNESS)
    mel.recompile_material(m)
    return m


def dust():
    m = material("M_RubbleDust")
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_VOLUMETRIC_NON_DIRECTIONAL)
    mel.connect_material_property(vector(m, "Color", (0.55, 0.5, 0.44, 1.0), -600, -200), "", MP.MP_BASE_COLOR)
    fade = expr(m, unreal.MaterialExpressionPerInstanceCustomData, -900, 100, data_index=3, const_default_value=1.0)
    # Soft edges: fresnel against the view, so a sphere reads as a puff.
    fresnel = expr(m, unreal.MaterialExpressionFresnel, -900, 250)
    one = const(m, 1.0, -900, 400)
    soft = binary(m, unreal.MaterialExpressionSubtract, one, fresnel, -700, 300)
    puff = noise(m, 0.03, -900, 550, lo=0.4, hi=1.0, levels=2)
    a = binary(m, unreal.MaterialExpressionMultiply, soft, puff, -550, 350)
    b = binary(m, unreal.MaterialExpressionMultiply, a, fade, -400, 250)
    c = binary(m, unreal.MaterialExpressionMultiply, b, scalar(m, "Density", 0.16, -550, 500), -250, 300)
    mel.connect_material_property(c, "", MP.MP_OPACITY)
    mel.recompile_material(m)
    return m


def spark():
    m = material("M_RubbleSpark")
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_ADDITIVE)
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    glow = expr(m, unreal.MaterialExpressionPerInstanceCustomData, -700, 100, data_index=3, const_default_value=1.0)
    color = vector(m, "Glow", (30.0, 10.0, 1.5, 1.0), -700, -100)
    mel.connect_material_property(binary(m, unreal.MaterialExpressionMultiply, color, glow, -400, 0), "", MP.MP_EMISSIVE_COLOR)
    mel.recompile_material(m)
    return m


def nanite_meshes():
    """Nanite copies of the engine's basic shapes (runtime-built meshes cannot be Nanite)."""
    root = "/Game/Rubble/Meshes"
    if not eal.does_directory_exist(root):
        eal.make_directory(root)
    for src, name in [("/Engine/BasicShapes/Cube", "SM_NaniteBox"), ("/Engine/BasicShapes/Sphere", "SM_NaniteSphere")]:
        dst = f"{root}/{name}"
        if eal.does_asset_exist(dst):
            eal.delete_asset(dst)
        mesh = eal.duplicate_asset(src, dst)
        settings = mesh.get_editor_property("nanite_settings")
        settings.set_editor_property("enabled", True)
        mesh.set_editor_property("nanite_settings", settings)
        eal.save_asset(dst, only_if_is_dirty=False)
    log("nanite meshes: SM_NaniteBox, SM_NaniteSphere")


def main():
    if not eal.does_directory_exist(ROOT):
        eal.make_directory(ROOT)
    s = surface()
    instance("MI_Concrete", s, {"Roughness": 0.85, "Grit": 0.35}, {"BaseColor": (0.5, 0.49, 0.46, 1)})
    instance("MI_Brick", s, {"Roughness": 0.8, "Grit": 0.4}, {"BaseColor": (0.45, 0.17, 0.11, 1)})
    instance("MI_Timber", s, {"Roughness": 0.7, "Grit": 0.45}, {"BaseColor": (0.55, 0.38, 0.2, 1)})
    instance("MI_Steel", s, {"Roughness": 0.35, "Metallic": 1.0, "Grit": 0.2}, {"BaseColor": (0.6, 0.6, 0.62, 1)})
    instance("MI_Glass", s, {"Roughness": 0.03, "Specular": 1.0, "Grit": 0.0}, {"BaseColor": (0.02, 0.025, 0.03, 1)})
    instance("MI_Rubber", s, {"Roughness": 0.9, "Grit": 0.15}, {"BaseColor": (0.025, 0.025, 0.025, 1)})
    instance("MI_Plastic", s, {"Roughness": 0.45, "Grit": 0.1}, {"BaseColor": (0.6, 0.6, 0.6, 1)})
    instance("MI_Interior", s, {"Roughness": 0.95, "Grit": 0.6}, {"BaseColor": (0.36, 0.34, 0.31, 1)})
    car_paint()
    g = ground()
    instance("MI_GroundYard", g, {"Roughness": 0.85}, {"BaseColor": (0.06, 0.06, 0.065, 1)})
    instance("MI_GroundHall", g, {"Roughness": 0.7}, {"BaseColor": (0.32, 0.32, 0.31, 1)})
    instance("MI_GroundLot", g, {"Roughness": 0.9}, {"BaseColor": (0.09, 0.085, 0.08, 1)})
    dust()
    spark()
    nanite_meshes()
    eal.save_directory(ROOT, only_if_is_dirty=False, recursive=True)
    log("done: " + ", ".join(sorted(a.split("/")[-1].split(".")[0] for a in eal.list_assets(ROOT))))


main()
