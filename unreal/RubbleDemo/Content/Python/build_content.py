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
  M_RubbleDust      sun-lit translucent dust puffs: tint from custom data 0..2, opacity from 3,
                    soft billowing edges, depth-faded against geometry.
  M_RubbleSpark     additive sparks, glow from custom data 3.
  M_RubbleTownChunk the town's chunks: vibe-land's 2K Poly Haven texture arrays (CC0, copied by
                    scripts/unreal/fetch-city-textures.sh), projected on each face's dominant axis
                    in the chunk's own frame offset by its place in the town, so an intact wall is
                    one surface and a broken piece keeps its texture. Per-instance custom data:
                    0..2 tint (the material's colour), 3 layer (-1: tint only), 4 tiles per metre,
                    5..7 the chunk's position in its destructible (cm), 8..10 the instance's scale.
                    Layer is a code: wall layer + 16 x up-facing layer; 0..2 then multiply the
                    texture (the material's colour over the layer's mean), 11 is the up layer's
                    tiles per metre and 12..15 the chunk's rotation in its destructible. Broad weathering and ground grime run across chunks.
  M_RubbleTownGlass the town's glass: a dark reflective pane over rooms traced behind it (interior
                    mapping), randomly lit. Layer code + 512 marks a glazed facade bay (the towers'
                    walls), which the chunk material draws as a framed window over the same rooms.
  M_RubbleTownGround grass and muddy leaves from the same arrays, for the ground around the town.

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


# The soft car (the engine's procedural car mesh, skinned each frame): vertex colour R is the
# crumple strain the skinning kernel measured, G the panel seam; UV0 and UV1.x hold the rest
# position in metres (UV0 = x, y; UV1.x = z), so damage patterns stay on the panel as it bends.
SOFT_PAINT = """
float d = smoothstep(0.015, 0.25, Strain);
float bare = saturate(smoothstep(0.86, 0.9, Streak) * smoothstep(0.25, 0.8, d) + smoothstep(0.55, 1.0, d) * step(0.55, Blotch));
float3 base = lerp(Paint * (1.0 - 0.45 * d), float3(0.5, 0.51, 0.53), bare);
Metal = bare;
Rough = lerp(lerp(0.3, 0.55, d), 0.38, bare);
Coat = (1.0 - d) * (1.0 - bare);
if (Side < 0.0) { base = float3(0.18, 0.18, 0.17); Metal = 0.0; Rough = 0.8; Coat = 0.0; }
base *= 1.0 - 0.85 * smoothstep(0.82, 0.97, Seam);
return base;
"""

SOFT_GLASS = """
float crack = smoothstep(0.012, 0.05, Strain) * (1.0 - smoothstep(0.0, 0.018, abs(Web - 0.5)));
Opacity = saturate(0.32 + 0.6 * Fres + crack);
Rough = lerp(0.02, 0.5, crack);
return lerp(float3(0.015, 0.02, 0.025), float3(0.7, 0.72, 0.74), crack);
"""


def rest_position(m, x, y):
    """The vertex's rest position in metres (UV0 = x, y; UV1.x = z)."""
    uv0 = expr(m, unreal.MaterialExpressionTextureCoordinate, x - 300, y, coordinate_index=0)
    uv1 = expr(m, unreal.MaterialExpressionTextureCoordinate, x - 300, y + 80, coordinate_index=1)
    mask = expr(m, unreal.MaterialExpressionComponentMask, x - 150, y + 80, r=True, g=False, b=False, a=False)
    link(uv1, "", mask, "")
    return binary(m, unreal.MaterialExpressionAppendVector, uv0, mask, x, y)


def rest_noise(m, rest, scale, x, y):
    """Noise of the rest position scaled per axis (scale: x, y, z), 0..1."""
    k = expr(m, unreal.MaterialExpressionConstant3Vector, x - 300, y, constant=unreal.LinearColor(scale[0], scale[1], scale[2], 0.0))
    p = binary(m, unreal.MaterialExpressionMultiply, rest, k, x - 150, y)
    n = expr(m, unreal.MaterialExpressionNoise, x, y, scale=1.0, levels=1, output_min=0.0, output_max=1.0, quality=1)
    link(p, "", n, "Position")
    return n


def soft_car():
    """The soft car's surfaces (RubbleRender::SoftCarSurface; car-render.mjs is the web's).

    M_RubbleSoftCarPaint: clear-coat paint that dulls and darkens where the panel crumpled and is
    scraped through to bare steel where it crumpled most; grey primer on a panel's inside (seen
    through tears); dark panel gaps along seams. M_RubbleSoftCarGlass: dark tinted glass, Fresnel
    opacity, a web of cracks as the pane strains. M_RubbleSoftCarLamp: lamps that go out when
    crushed. The rest are M_RubbleSurface instances (chrome, trim, underbody, interior, engine,
    grille)."""
    s = eal.load_asset(f"{ROOT}/M_RubbleSurface")
    m = material("M_RubbleSoftCarPaint")
    m.set_editor_property("used_with_instanced_static_meshes", False)
    m.set_editor_property("two_sided", True)
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_CLEAR_COAT)
    vc = expr(m, unreal.MaterialExpressionVertexColor, -1500, -300)
    rest = rest_position(m, -1500, 100)
    c = custom(m, SOFT_PAINT, -800, 0, {
        "Paint": (vector(m, "Paint", (0.05, 0.18, 0.55, 1.0), -1500, -450), ""),
        "Strain": (vc, "R"),
        "Seam": (vc, "G"),
        "Side": (expr(m, unreal.MaterialExpressionTwoSidedSign, -1500, -150), ""),
        "Streak": (rest_noise(m, rest, (6.0, 90.0, 90.0), -1100, 250), ""),
        "Blotch": (rest_noise(m, rest, (18.0, 18.0, 18.0), -1100, 400), ""),
    }, {"Metal": 1, "Rough": 1, "Coat": 1}, "SoftCarPaint")
    # Clear coat goes in through material attributes (its custom-data pins are hidden from Python).
    m.set_editor_property("use_material_attributes", True)
    attrs = expr(m, unreal.MaterialExpressionMakeMaterialAttributes, -450, 0)
    link(c, "", attrs, "BaseColor")
    link(c, "Metal", attrs, "Metallic")
    link(c, "Rough", attrs, "Roughness")
    link(c, "Coat", attrs, "ClearCoat")
    link(const(m, 0.06, -800, 300), "", attrs, "ClearCoatRoughness")
    mel.connect_material_property(attrs, "", MP.MP_MATERIAL_ATTRIBUTES)
    mel.recompile_material(m)

    g = material("M_RubbleSoftCarGlass")
    g.set_editor_property("used_with_instanced_static_meshes", False)
    g.set_editor_property("two_sided", True)
    g.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    g.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING)
    vc = expr(g, unreal.MaterialExpressionVertexColor, -1500, -300)
    rest = rest_position(g, -1500, 100)
    web = rest_noise(g, rest, (22.0, 22.0, 22.0), -1100, 250)
    fres = expr(g, unreal.MaterialExpressionFresnel, -1100, -100, exponent=4.0, base_reflect_fraction=0.04)
    c = custom(g, SOFT_GLASS, -800, 0, {"Strain": (vc, "R"), "Web": (web, ""), "Fres": (fres, "")}, {"Opacity": 1, "Rough": 1}, "SoftCarGlass")
    mel.connect_material_property(c, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(c, "Opacity", MP.MP_OPACITY)
    mel.connect_material_property(c, "Rough", MP.MP_ROUGHNESS)
    mel.connect_material_property(const(g, 1.0, -800, 300), "", MP.MP_SPECULAR)
    mel.recompile_material(g)

    lamp = material("M_RubbleSoftCarLamp")
    lamp.set_editor_property("used_with_instanced_static_meshes", False)
    color = vector(lamp, "Color", (1.0, 0.95, 0.85, 1.0), -900, -200)
    vc = expr(lamp, unreal.MaterialExpressionVertexColor, -900, 100)
    out = expr(lamp, unreal.MaterialExpressionSmoothStep, -700, 100)
    link(const(lamp, 0.3, -900, 220), "", out, "Min")
    link(const(lamp, 0.05, -900, 280), "", out, "Max")
    link(vc, "R", out, "Value")
    glow = binary(lamp, unreal.MaterialExpressionMultiply, color, scalar(lamp, "Intensity", 8.0, -900, 0), -550, -100)
    mel.connect_material_property(binary(lamp, unreal.MaterialExpressionMultiply, glow, out, -350, 0), "", MP.MP_EMISSIVE_COLOR)
    mel.connect_material_property(binary(lamp, unreal.MaterialExpressionMultiply, color, const(lamp, 0.3, -550, -250), -350, -250), "", MP.MP_BASE_COLOR)
    mel.connect_material_property(const(lamp, 0.08, -350, 150), "", MP.MP_ROUGHNESS)
    mel.recompile_material(lamp)
    instance("MI_SoftCarHeadlight", lamp, {"Intensity": 6.0}, {"Color": (1.0, 0.95, 0.85, 1)})
    instance("MI_SoftCarTaillight", lamp, {"Intensity": 4.0}, {"Color": (0.9, 0.02, 0.015, 1)})
    instance("MI_SoftCarChrome", s, {"Roughness": 0.06, "Metallic": 1.0, "Specular": 1.0, "Grit": 0.0}, {"BaseColor": (0.92, 0.92, 0.95, 1)})
    instance("MI_SoftCarTrim", s, {"Roughness": 0.55, "Grit": 0.1}, {"BaseColor": (0.025, 0.025, 0.028, 1)})
    instance("MI_SoftCarUnderbody", s, {"Roughness": 0.85, "Grit": 0.5}, {"BaseColor": (0.035, 0.035, 0.035, 1)})
    instance("MI_SoftCarInterior", s, {"Roughness": 0.9, "Grit": 0.3}, {"BaseColor": (0.045, 0.045, 0.05, 1)})
    instance("MI_SoftCarEngine", s, {"Roughness": 0.4, "Metallic": 0.7, "Grit": 0.4}, {"BaseColor": (0.22, 0.22, 0.23, 1)})
    instance("MI_SoftCarGrille", s, {"Roughness": 0.5, "Grit": 0.6}, {"BaseColor": (0.02, 0.02, 0.02, 1)})


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
    """M_RubbleDust: a puff of dust on a sphere instance (DUST, defined after TOWN_NOISE): tinted
    from custom data 0..2, opacity faded by 3; soft billowing edges from noise in the instance's
    own (randomly turned) space, faded where it meets geometry, lit by the sun through the
    translucency volume."""
    m = material("M_RubbleDust")
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_VOLUMETRIC_DIRECTIONAL)
    data = [expr(m, unreal.MaterialExpressionPerInstanceCustomData, -1300, -400 + 90 * i, data_index=i, const_default_value=(1.0 if i == 3 else 0.5)) for i in range(4)]
    rg = binary(m, unreal.MaterialExpressionAppendVector, data[0], data[1], -1100, -380)
    tint = binary(m, unreal.MaterialExpressionAppendVector, rg, data[2], -950, -360)
    mel.connect_material_property(tint, "", MP.MP_BASE_COLOR)
    world = expr(m, unreal.MaterialExpressionWorldPosition, -1300, 0)
    local_p = expr(m, unreal.MaterialExpressionTransformPosition, -1100, 0,
                   transform_source_type=unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_WORLD,
                   transform_type=unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_INSTANCE)
    link(world, "", local_p, "")
    inputs = {
        "LocalP": (local_p, ""),
        "Nw": (expr(m, unreal.MaterialExpressionVertexNormalWS, -1300, 150), ""),
        "V": (expr(m, unreal.MaterialExpressionCameraVectorWS, -1300, 250), ""),
        "T": (expr(m, unreal.MaterialExpressionTime, -1300, 350), ""),
        "Fade": (data[3], ""),
        "Density": (scalar(m, "Density", 1.1, -1300, 450), ""),
    }
    c = custom(m, DUST, -800, 100, inputs, {}, "Dust")
    fade = expr(m, unreal.MaterialExpressionDepthFade, -500, 150)
    fade.set_editor_property("fade_distance_default", 120.0)
    link(c, "", fade, "Opacity")
    mel.connect_material_property(fade, "", MP.MP_OPACITY)
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


# Value noise and a hash, for the town's Custom nodes (HLSL has no free functions there, so they
# live on a struct).
TOWN_NOISE = """
struct FTownNoise
{
    float H(float2 q) { return frac(sin(dot(q, float2(127.1, 311.7))) * 43758.5453); }
    float H3(float3 q) { return frac(sin(dot(q, float3(12.9898, 78.233, 37.719))) * 43758.5453); }
    float V(float2 q)
    {
        float2 i = floor(q);
        float2 f = frac(q);
        f = f * f * (3.0 - 2.0 * f);
        return lerp(lerp(H(i), H(i + float2(1, 0)), f.x), lerp(H(i + float2(0, 1)), H(i + float2(1, 1)), f.x), f.y);
    }
    // Rooms behind a pane (interior mapping): the view ray from the pane at P (normal Nw, toward
    // the camera V) is traced through a grid of rooms that starts at the pane along its facing
    // axis, so the first room is right behind it. Each room is randomly lit, its walls, floor and
    // ceiling shaded, darker in the corners and further back. Returns the room's light (for
    // Emissive); Fres is the pane's Fresnel term.
    float3 Interior(float3 P, float3 Nw, float3 V, float Room, float Storey, float Brightness, out float Fres)
    {
        float3 d = -V;
        float3 an = abs(Nw);
        float3 mask = (an.x >= an.y && an.x >= an.z) ? float3(1, 0, 0) : (an.y >= an.z ? float3(0, 1, 0) : float3(0, 0, 1));
        float3 room = float3(Room, Room, Storey);
        float3 origin = P * mask;
        float3 p = P - origin + d * 0.5;
        d = d + (step(0.0, d) * 2.0 - 1.0) * 1e-4;
        float3 cell = floor(p / room);
        float3 lo = cell * room;
        float3 tt = (lo + step(0.0, d) * room - p) / d;
        float tm = min(tt.x, min(tt.y, tt.z));
        float3 q = (p + d * tm - lo) / room;
        // The room's place along the facade, and the facade's own only coarsely: pieces of a
        // fractured wall sit a few cm apart in depth and must show the same rooms.
        float seed = H3(cell * (1.0 - mask) + floor(origin / 1000.0 + 0.5));
        float3 col = lerp(float3(0.62, 0.56, 0.47), float3(0.45, 0.5, 0.55), frac(seed * 7.31));
        if (tm == tt.z)
        {
            // Floor, or a ceiling lit only from the room (darker than its walls).
            col = d.z < 0.0 ? float3(0.3, 0.2, 0.12) : float3(0.4, 0.39, 0.37);
        }
        float3 c = abs(q - 0.5) * 2.0;
        float edge = max(max(c.x * (1.0 - mask.x), c.y * (1.0 - mask.y)), c.z);
        col *= 0.55 + 0.45 * (1.0 - pow(saturate(edge), 6.0));
        // About half the rooms have their lights on (warm); the rest show cool daylight.
        float lit = seed > 0.5 ? 1.0 : 0.25;
        float3 light = lerp(float3(0.75, 0.8, 0.9), float3(1.0, 0.82, 0.62), step(0.5, seed));
        // A blind part-way down some panes.
        float2 pane = (mask.x > 0.5) ? P.yz : (mask.y > 0.5 ? P.xz : P.xy);
        float blind = step(0.72, frac(seed * 13.7)) * step(1.0 - 0.6 * frac(seed * 3.1), frac(pane.y / Storey));
        col = lerp(col, float3(0.5, 0.47, 0.42), blind);
        Fres = 0.04 + 0.96 * pow(1.0 - saturate(dot(Nw, V)), 5.0);
        return col * light * lit * Brightness * (1.0 - Fres) * exp(-tm / 900.0);
    }
};
FTownNoise N;
"""


DUST = TOWN_NOISE + """
float facing = saturate(dot(normalize(Nw), normalize(V)));
float edge = pow(facing, 1.4);
float3 q = LocalP / 50.0;
float t = T * 0.06;
float n1 = N.V(q.xy * 2.1 + t) * 0.5 + N.V(q.yz * 4.3 - t * 1.3) * 0.3 + N.V(q.zx * 8.7 + t * 2.1) * 0.2;
float billow = smoothstep(0.2, 0.8, n1 * 0.9 + edge * 0.6 - 0.15) * edge;
return saturate(billow * Fade * Density);
"""


def town_means():
    """Each texture-array layer's mean albedo (linear) as an HLSL table, from texture-means.json
    (written by scripts/unreal/fetch-city-textures.sh); mid grey when it is missing."""
    import json
    import os
    path = os.path.join(unreal.Paths.project_content_dir(), "Rubble", "Town", "texture-means.json")
    means = json.load(open(path))["meanLinear"] if os.path.exists(path) else [[0.5, 0.5, 0.5]] * 16
    rows = ", ".join(f"float3({r:.4f}, {g:.4f}, {b:.4f})" for r, g, b in means)
    return f"const float3 TOWN_MEANS[{len(means)}] = {{ {rows} }};\n"


TOWN_CHUNK = TOWN_NOISE + """
// The chunk's surface in its destructible's frame: the instance's local frame turned by the
// chunk's rotation there (custom data 12..15) and offset by its place. A shared hull placed turned
// then projects as its neighbours do.
float3 qv = Rotation.xyz;
float qw = Rotation.w;
float3 lp = LocalP * Scale;
float3 n0 = normalize(LocalN);
float3 n = n0 + 2.0 * cross(qv, cross(qv, n0) + qw * n0);
float3 p = lp + 2.0 * cross(qv, cross(qv, lp) + qw * lp) + Offset;
float3 a = abs(n);
float2 uv;
float3 t, b;
if (a.x >= a.y && a.x >= a.z) { uv = p.yz; t = float3(0, 1, 0); b = float3(0, 0, 1); }
else if (a.y >= a.z) { uv = p.xz; t = float3(1, 0, 0); b = float3(0, 0, 1); }
else { uv = p.xy; t = float3(1, 0, 0); b = float3(0, 1, 0); }
// The face's place in the destructible (cm), before tiling: the glazing's grid.
float2 face = uv;
// Faces that look up (slab tops, roof slopes, broken edges) take the up layer.
float3 nw = normalize(WorldN);
bool up = nw.z > 0.55;
// Glazed: the facades the piece reaches, as bits (1 +X, 2 -X, 4 +Y, 8 -Y in its destructible).
float glazed = floor(Layer / 512.0 + 0.001);
float code = Layer - 512.0 * glazed;
float untintedUp = floor(code / 256.0 + 0.001);
code -= 256.0 * untintedUp;
float3 albedo = Tint;
float3 nl = n;
float rough = 0.8;
float metal = 0.0;
float ao = 1.0;
Emissive = float3(0, 0, 0);
if (Layer > -0.5)
{
    float upLayer = floor(code / 16.0 + 0.001);
    float wallLayer = code - 16.0 * upLayer;
    float layer = up ? upLayer : wallLayer;
    uv = float2(uv.x, -uv.y) * ((up ? UpTiles : Tiles) / 100.0);
    float3 uvw = float3(uv, layer);
    albedo = Texture2DArraySample(Albedo, AlbedoSampler, uvw).rgb;
    // Tinted to the material's colour (custom data 0..2): the texture times colour over the
    // layer's mean, bounded, so it keeps its detail and averages to the colour.
    float3 ratio = clamp(Tint / max(TOWN_MEANS[int(layer)], 1e-3), 0.25, 3.0);
    albedo *= lerp(float3(1, 1, 1), ratio, (up && untintedUp > 0.5) ? 0.0 : TintStrength);
    // Normal-map compression keeps X and Y only (BC5): rebuild Z.
    float2 nxy = Texture2DArraySample(NormalMap, NormalMapSampler, uvw).xy * 2.0 - 1.0;
    float3 tn = float3(nxy, sqrt(saturate(1.0 - dot(nxy, nxy))));
    // DirectX normal maps: green points down the image, which is -b here.
    nl = normalize(t * tn.x - b * tn.y + n * tn.z);
    float3 arm = Texture2DArraySample(Arm, ArmSampler, uvw).rgb;
    ao = arm.r;
    rough = arm.g;
    metal = arm.b;
}
// Weathering in town space, so it runs across chunks: broad patches of tone and roughness, and
// grime rising from the ground on walls.
float macro = N.V(WorldP.xy / 900.0) * 0.6 + N.V(WorldP.xy / 230.0 + 17.0) * 0.4;
albedo *= lerp(1.0 - Weathering, 1.0 + 0.5 * Weathering, macro);
rough = saturate(rough + (macro - 0.5) * 0.15);
if (!up)
{
    float grime = 1.0 - saturate(WorldP.z / 140.0);
    albedo *= 1.0 - Grime * grime * grime;
}
// A glazed facade (a tower's walls): framed windows on a grid of bays and storeys in the
// destructible's frame, so each window moves and breaks with its chunk, with rooms behind them.
// Only faces pointing out of a facade the piece reaches: not a fracture facet or a recess.
float4 bits = fmod(floor(glazed / float4(1, 2, 4, 8)), 2.0);
float outward = max(max(bits.x * n.x, bits.y * -n.x), max(bits.z * n.y, bits.w * -n.y));
if (glazed > 0.5 && outward > 0.985)
{
    float2 g = float2(frac(face.x / WindowBay), frac(face.y / StoreyHeight));
    float2 w = abs(g - float2(0.5, 0.58));
    float inside = step(w.x, 0.4) * step(w.y, 0.3);
    float frame = step(w.x, 0.44) * step(w.y, 0.34) - inside;
    if (inside > 0.5)
    {
        float fres;
        Emissive = N.Interior(WorldP, nw, normalize(CameraVec), RoomSize, StoreyHeight, Brightness, fres);
        albedo = float3(0.012, 0.016, 0.02);
        nl = n;
        rough = PaneRoughness;
        metal = 0.0;
        ao = 1.0;
    }
    else if (frame > 0.5)
    {
        albedo = float3(0.05, 0.055, 0.06);
        rough = 0.35;
        metal = 0.8;
    }
}
// Back to the instance's frame, which the material's normal is in.
Normal = nl - 2.0 * cross(qv, qw * nl - cross(qv, nl));
Roughness = rough;
Metallic = metal;
Occlusion = ao;
return albedo;
"""

# Glass over rooms (FTownNoise::Interior), with Lumen reflecting the street in the pane on top.
TOWN_GLASS = TOWN_NOISE + """
float fres;
float3 light = N.Interior(WorldP, normalize(WorldN), normalize(CameraVec), RoomSize, StoreyHeight, Brightness, fres);
Fresnel = fres;
return light;
"""

# The open ground around the town: leafy grass, broken by patches of muddy leaves.
TOWN_GROUND = TOWN_NOISE + """
float2 uv = WorldP.xy / 100.0;
float m = N.V(WorldP.xy / 1400.0) * 0.65 + N.V(WorldP.xy / 380.0 + 9.0) * 0.35;
float mud = smoothstep(0.66, 0.8, m) * 0.85;
float3 g = Texture2DArraySample(Albedo, AlbedoSampler, float3(uv / GrassMetres, GrassLayer)).rgb;
float3 u = Texture2DArraySample(Albedo, AlbedoSampler, float3(uv / MudMetres, MudLayer)).rgb;
float2 gn = Texture2DArraySample(NormalMap, NormalMapSampler, float3(uv / GrassMetres, GrassLayer)).xy * 2.0 - 1.0;
float2 un = Texture2DArraySample(NormalMap, NormalMapSampler, float3(uv / MudMetres, MudLayer)).xy * 2.0 - 1.0;
float2 nxy = lerp(gn, un, mud);
Normal = normalize(float3(nxy.x, -nxy.y, sqrt(saturate(1.0 - dot(nxy, nxy)))));
float3 ga = Texture2DArraySample(Arm, ArmSampler, float3(uv / GrassMetres, GrassLayer)).rgb;
float3 ua = Texture2DArraySample(Arm, ArmSampler, float3(uv / MudMetres, MudLayer)).rgb;
float3 arm = lerp(ga, ua, mud);
Roughness = arm.g;
Occlusion = arm.r;
// The grass photograph is ochre: vibe-land's canopy tint turns it green, less so in the dry patches.
float3 canopy = lerp(CanopyTint.rgb, float3(1, 1, 1), smoothstep(0.35, 0.6, m) * 0.5);
return lerp(g * canopy, u, mud) * lerp(0.85, 1.1, m);
"""


def city_arrays():
    """vibe-land's HD texture arrays, or None when scripts/unreal/fetch-city-textures.sh has not run."""
    arrays = {k: unreal.load_asset(f"/Game/VibeCity/Textures/T_City{k}HDArray") for k in ("Albedo", "Normal", "ARM")}
    return arrays if all(arrays.values()) else None


def custom(m, code, x, y, inputs, outputs, description):
    """A Custom node fed by inputs ({name: (expression, pin)}) with extra float outputs ({name: width})."""
    widths = {1: unreal.CustomMaterialOutputType.CMOT_FLOAT1, 3: unreal.CustomMaterialOutputType.CMOT_FLOAT3}
    c = expr(m, unreal.MaterialExpressionCustom, x, y, code=code, output_type=unreal.CustomMaterialOutputType.CMOT_FLOAT3, description=description)
    ins = []
    for name in inputs:
        ci = unreal.CustomInput()
        ci.set_editor_property("input_name", name)
        ins.append(ci)
    c.set_editor_property("inputs", ins)
    outs = []
    for name, width in outputs.items():
        co = unreal.CustomOutput()
        co.set_editor_property("output_name", name)
        co.set_editor_property("output_type", widths[width])
        outs.append(co)
    c.set_editor_property("additional_outputs", outs)
    for name, (source, pin) in inputs.items():
        link(source, pin, c, name)
    return c


def texture_objects(m, arrays, x, y):
    out = {}
    for i, (key, name) in enumerate((("Albedo", "Albedo"), ("Normal", "NormalMap"), ("ARM", "Arm"))):
        o = expr(m, unreal.MaterialExpressionTextureObject, x, y + 130 * i)
        o.set_editor_property("texture", arrays[key])
        out[name] = (o, "")
    return out


def world_normal(m, x, y):
    return expr(m, unreal.MaterialExpressionVertexNormalWS, x, y)


def town_chunk(arrays):
    """M_RubbleTownChunk (see the module docstring)."""
    m = material("M_RubbleTownChunk")
    m.set_editor_property("tangent_space_normal", False)
    defaults = {3: -1.0, 4: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 15: 1.0}
    data = [expr(m, unreal.MaterialExpressionPerInstanceCustomData, -1800, -700 + 90 * i, data_index=i, const_default_value=defaults.get(i, 0.0)) for i in range(16)]

    def append3(i, x, y):
        ab = binary(m, unreal.MaterialExpressionAppendVector, data[i], data[i + 1], x, y)
        return binary(m, unreal.MaterialExpressionAppendVector, ab, data[i + 2], x + 120, y)

    append4 = binary(m, unreal.MaterialExpressionAppendVector, append3(12, -1600, 1540), data[15], -1360, 1540)
    world = expr(m, unreal.MaterialExpressionWorldPosition, -1800, 150)
    # Instance space: an instanced mesh's Local space is its component's, not the instance's.
    local_p = expr(m, unreal.MaterialExpressionTransformPosition, -1600, 150,
                   transform_source_type=unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_WORLD,
                   transform_type=unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_INSTANCE)
    link(world, "", local_p, "")
    normal_ws = world_normal(m, -1800, 300)
    local_n = expr(m, unreal.MaterialExpressionTransform, -1600, 300,
                   transform_source_type=unreal.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_WORLD,
                   transform_type=unreal.MaterialVectorCoordTransform.TRANSFORM_INSTANCE)
    link(normal_ws, "", local_n, "")
    inputs = {
        "LocalP": (local_p, ""), "LocalN": (local_n, ""), "WorldP": (world, ""), "WorldN": (normal_ws, ""),
        "Scale": (append3(8, -1600, -100), ""), "Offset": (append3(5, -1600, -250), ""),
        "Rotation": (append4, ""),
        "Tiles": (data[4], ""), "UpTiles": (data[11], ""), "Layer": (data[3], ""), "Tint": (append3(0, -1600, -700), ""),
        "TintStrength": (scalar(m, "TintStrength", 0.8, -1600, 900), ""),
        "Weathering": (scalar(m, "Weathering", 0.18, -1600, 980), ""),
        "Grime": (scalar(m, "Grime", 0.35, -1600, 1060), ""),
        "CameraVec": (expr(m, unreal.MaterialExpressionCameraVectorWS, -1600, 1140), ""),
        "RoomSize": (scalar(m, "RoomSize", 420.0, -1600, 1220), ""),
        "StoreyHeight": (scalar(m, "StoreyHeight", 306.0, -1600, 1300), ""),
        "Brightness": (scalar(m, "Brightness", 0.45, -1600, 1380), ""),
        # The towers' bays (54.6 m of perimeter in 40 wall pieces) and storeys (30.6 m in ten).
        "WindowBay": (scalar(m, "WindowBay", 137.0, -1600, 1460), ""),
        "PaneRoughness": (scalar(m, "PaneRoughness", 0.3, -1600, 1620), ""),
    }
    inputs.update(texture_objects(m, arrays, -1600, 450))
    c = custom(m, town_means() + TOWN_CHUNK, -1100, 0, inputs, {"Normal": 3, "Roughness": 1, "Metallic": 1, "Occlusion": 1, "Emissive": 3}, "TownChunk")
    to_world = expr(m, unreal.MaterialExpressionTransform, -700, 150,
                    transform_source_type=unreal.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_INSTANCE,
                    transform_type=unreal.MaterialVectorCoordTransform.TRANSFORM_WORLD)
    link(c, "Normal", to_world, "")
    normal = expr(m, unreal.MaterialExpressionNormalize, -500, 150)
    link(to_world, "", normal, "")
    mel.connect_material_property(c, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(normal, "", MP.MP_NORMAL)
    mel.connect_material_property(c, "Roughness", MP.MP_ROUGHNESS)
    mel.connect_material_property(c, "Metallic", MP.MP_METALLIC)
    mel.connect_material_property(c, "Occlusion", MP.MP_AMBIENT_OCCLUSION)
    mel.connect_material_property(c, "Emissive", MP.MP_EMISSIVE_COLOR)
    mel.recompile_material(m)
    return m


def town_glass():
    """M_RubbleTownGlass: dark reflective panes over lit rooms (TOWN_GLASS)."""
    m = material("M_RubbleTownGlass")
    inputs = {
        "WorldP": (expr(m, unreal.MaterialExpressionWorldPosition, -1400, -200), ""),
        "WorldN": (world_normal(m, -1400, -50), ""),
        "CameraVec": (expr(m, unreal.MaterialExpressionCameraVectorWS, -1400, 100), ""),
        "RoomSize": (scalar(m, "RoomSize", 420.0, -1400, 250), ""),
        "StoreyHeight": (scalar(m, "StoreyHeight", 300.0, -1400, 330), ""),
        "Brightness": (scalar(m, "Brightness", 0.35, -1400, 410), ""),
    }
    c = custom(m, TOWN_GLASS, -900, 0, inputs, {"Fresnel": 1}, "TownGlass")
    mel.connect_material_property(vector(m, "PaneColor", (0.012, 0.016, 0.02, 1.0), -600, -250), "", MP.MP_BASE_COLOR)
    mel.connect_material_property(c, "", MP.MP_EMISSIVE_COLOR)
    mel.connect_material_property(scalar(m, "Roughness", 0.03, -600, 200), "", MP.MP_ROUGHNESS)
    mel.connect_material_property(scalar(m, "Specular", 0.5, -600, 280), "", MP.MP_SPECULAR)
    mel.recompile_material(m)
    return m


def curtain_glass():
    """M_RubbleCurtainGlass: a glass tower's curtain wall (the pack's curtain-glass), seen
    through. Translucent and lit per pixel (surface forward shading), so the sun's highlight is
    sharp and Lumen's front-layer reflections reach it; Fresnel opacity: clear face on, a mirror
    at grazing angles, as float glass is; a faint green-blue body tint."""
    m = material("M_RubbleCurtainGlass")
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING)
    fres = expr(m, unreal.MaterialExpressionFresnel, -900, 200, exponent=3.0, base_reflect_fraction=0.04)
    opacity = expr(m, unreal.MaterialExpressionLinearInterpolate, -600, 200)
    mel.connect_material_expressions(scalar(m, "FaceOpacity", 0.38, -900, 80), "", opacity, "A")
    mel.connect_material_expressions(scalar(m, "EdgeOpacity", 0.92, -900, 140), "", opacity, "B")
    mel.connect_material_expressions(fres, "", opacity, "Alpha")
    mel.connect_material_property(opacity, "", MP.MP_OPACITY)
    mel.connect_material_property(vector(m, "BodyTint", (0.035, 0.065, 0.08, 1.0), -600, -250), "", MP.MP_BASE_COLOR)
    mel.connect_material_property(scalar(m, "Roughness", 0.02, -600, 380), "", MP.MP_ROUGHNESS)
    mel.connect_material_property(scalar(m, "Specular", 1.0, -600, 450), "", MP.MP_SPECULAR)
    mel.recompile_material(m)
    return m


def town_ground(arrays):
    """M_RubbleTownGround: the grass and mud layers of the town's texture arrays, world-projected."""
    m = material("M_RubbleTownGround")
    m.set_editor_property("used_with_instanced_static_meshes", False)
    inputs = {
        "WorldP": (expr(m, unreal.MaterialExpressionWorldPosition, -1400, -200), ""),
        # The catalog's grass and mud-leaves layers (catalog.json) and their tile sizes.
        "GrassLayer": (const(m, 10.0, -1400, -50), ""), "GrassMetres": (const(m, 2.0, -1400, 20), ""),
        "MudLayer": (const(m, 11.0, -1400, 90), ""), "MudMetres": (const(m, 1.3, -1400, 160), ""),
        "CanopyTint": (vector(m, "CanopyTint", (0.5, 0.95, 0.36, 1.0), -1400, 200), ""),
    }
    inputs.update(texture_objects(m, arrays, -1400, 250))
    c = custom(m, TOWN_GROUND, -900, 0, inputs, {"Normal": 3, "Roughness": 1, "Occlusion": 1}, "TownGround")
    mel.connect_material_property(c, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(c, "Normal", MP.MP_NORMAL)
    mel.connect_material_property(c, "Roughness", MP.MP_ROUGHNESS)
    mel.connect_material_property(c, "Occlusion", MP.MP_AMBIENT_OCCLUSION)
    mel.recompile_material(m)
    return m


def town():
    """The town's materials. Glass needs no textures; the rest fall back to tint-only chunks and
    the yard's ground when the copied texture arrays are missing."""
    town_glass()
    curtain_glass()
    arrays = city_arrays()
    if not arrays:
        log("town: no /Game/VibeCity texture arrays (scripts/unreal/fetch-city-textures.sh); chunk and ground skipped")
        return
    town_chunk(arrays)
    town_ground(arrays)


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


# ----- Quixel Megascans (Assets/megascans.json; SourceArt/Megascans, git-ignored) -------------------------

import os, re, glob as _glob

MEGASCANS = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()), "SourceArt", "Megascans")
SCANS = "/Game/Megascans"


def scan_map(key, role):
    """The highest-resolution file of a role (BaseColor, Normal, Roughness, AO, Mask, Opacity) in a
    surface's folder, or None."""
    files = _glob.glob(os.path.join(MEGASCANS, key, f"*_{role}.jpg"))
    rank = lambda f: int((re.search(r"_(\d+)K_", f) or re.search(r"(\d+)", "0")).group(1))
    return max(files, key=rank) if files else None


def scan_size(key):
    """A surface's physical size (m), from its metadata (`physicalSize`: "2x2")."""
    for j in _glob.glob(os.path.join(MEGASCANS, key, "*.json")):
        m = re.search(r'"physicalSize":\s*"([\d.]+)x([\d.]+)"', open(j).read())
        if m:
            return float(m.group(1))
    return 2.0


def import_scan(key, role, kind):
    """One map into /Game/Megascans/<key>: colour (sRGB), normal, or linear grey (roughness, AO, masks)."""
    f = scan_map(key, role)
    if not f:
        return None
    dest = f"{SCANS}/{key}"
    name = os.path.splitext(os.path.basename(f))[0]
    path = f"{dest}/{name}"
    if not eal.does_asset_exist(path):
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", f)
        task.set_editor_property("destination_path", dest)
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("save", False)
        tools.import_asset_tasks([task])
    t = unreal.load_asset(path)
    if kind == "normal":
        t.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
        t.set_editor_property("srgb", False)
    elif kind == "grey":
        t.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_GRAYSCALE)
        t.set_editor_property("srgb", False)
    eal.save_loaded_asset(t)
    return t


def scan_set(key):
    """A surface's maps (Albedo, Normal, Roughness, AO) and size (m), or None when it is missing."""
    maps = {
        "Albedo": import_scan(key, "BaseColor", "color"),
        "Normal": import_scan(key, "Normal", "normal"),
        "Roughness": import_scan(key, "Roughness", "grey"),
        # Some scans carry no AO: their cavity map is the nearest.
        "AO": import_scan(key, "AO", "grey") or import_scan(key, "Cavity", "grey"),
    }
    if not all(maps.values()):
        return None
    return maps, scan_size(key)


def scan_missing(keys):
    manifest = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()), "Assets", "megascans.json")
    import json
    listing = {a["key"]: a for a in json.load(open(manifest))["assets"]}
    for k in keys:
        a = listing.get(k, {})
        log(f"megascans: missing {k} ({a.get('title', '?')}: https://www.fab.com/listings/{a.get('uid', '?')}); scripts/unreal/fetch-assets.sh")


def instance_space(m, x, y):
    """The pixel's position and normal in its instance's frame (cm): an instanced mesh's Local space
    is its component's, not the instance's."""
    world = expr(m, unreal.MaterialExpressionWorldPosition, x - 200, y)
    p = expr(m, unreal.MaterialExpressionTransformPosition, x, y,
             transform_source_type=unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_WORLD,
             transform_type=unreal.MaterialPositionTransformSource.TRANSFORMPOSSOURCE_INSTANCE)
    link(world, "", p, "")
    n = expr(m, unreal.MaterialExpressionTransform, x, y + 150,
             transform_source_type=unreal.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_WORLD,
             transform_type=unreal.MaterialVectorCoordTransform.TRANSFORM_INSTANCE)
    link(world_normal(m, x - 200, y + 150), "", n, "")
    return p, n


def normal_to_world(m, c, x, y):
    w = expr(m, unreal.MaterialExpressionTransform, x, y,
             transform_source_type=unreal.MaterialVectorCoordTransformSource.TRANSFORMSOURCE_INSTANCE,
             transform_type=unreal.MaterialVectorCoordTransform.TRANSFORM_WORLD)
    link(c, "Normal", w, "")
    n = expr(m, unreal.MaterialExpressionNormalize, x + 200, y)
    link(w, "", n, "")
    return n


def texture_param(m, name, default, sampler, x, y):
    return expr(m, unreal.MaterialExpressionTextureObjectParameter, x, y, parameter_name=name, texture=default, sampler_type=sampler)


SAMPLERS = {"Albedo": unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, "Normal": unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL,
            "Roughness": unreal.MaterialSamplerType.SAMPLERTYPE_GRAYSCALE, "AO": unreal.MaterialSamplerType.SAMPLERTYPE_GRAYSCALE}

# Triplanar in the instance's frame, three scans: Top on up faces, Bottom on down faces, Side on the
# rest, each at its scan's physical size; a broken piece keeps its texture as it tumbles. Normals blend
# per projection (each map's tangent frame is the projection's axes) and leave in the instance frame.
SCAN_TRIPLANAR = r"""
float3 n = normalize(N);
float3 w = pow(abs(n), 8.0); w /= (w.x + w.y + w.z);
float up = saturate(n.z * 1e4);
float2 uvX = P.yz / (SideSize * 100.0), uvY = P.xz / (SideSize * 100.0);
float2 uvT = P.xy / (TopSize * 100.0), uvB = P.xy / (BottomSize * 100.0);
float3 aX = Texture2DSample(SideAlbedo, SideAlbedoSampler, uvX).rgb;
float3 aY = Texture2DSample(SideAlbedo, SideAlbedoSampler, uvY).rgb;
float3 aZ = lerp(Texture2DSample(BottomAlbedo, BottomAlbedoSampler, uvB).rgb, Texture2DSample(TopAlbedo, TopAlbedoSampler, uvT).rgb, up);
float3 albedo = aX * w.x + aY * w.y + aZ * w.z;
// Large-scale variation over the tiles (the top scan at a seventh of its scale).
float3 big = Texture2DSample(TopAlbedo, TopAlbedoSampler, (P.xy + P.z * 0.37) / (TopSize * 100.0 * 7.3)).rgb;
albedo *= lerp(1.0, saturate(dot(big, float3(0.333, 0.333, 0.333)) / max(dot(aZ, float3(0.333, 0.333, 0.333)), 0.05)), Variation);
float rX = Texture2DSample(SideRoughness, SideRoughnessSampler, uvX).r, rY = Texture2DSample(SideRoughness, SideRoughnessSampler, uvY).r;
float rZ = lerp(Texture2DSample(BottomRoughness, BottomRoughnessSampler, uvB).r, Texture2DSample(TopRoughness, TopRoughnessSampler, uvT).r, up);
Roughness = saturate((rX * w.x + rY * w.y + rZ * w.z) * RoughnessScale);
float oX = Texture2DSample(SideAO, SideAOSampler, uvX).r, oY = Texture2DSample(SideAO, SideAOSampler, uvY).r;
float oZ = lerp(Texture2DSample(BottomAO, BottomAOSampler, uvB).r, Texture2DSample(TopAO, TopAOSampler, uvT).r, up);
Occlusion = oX * w.x + oY * w.y + oZ * w.z;
float2 tX = Texture2DSample(SideNormal, SideNormalSampler, uvX).xy * 2.0 - 1.0;
float2 tY = Texture2DSample(SideNormal, SideNormalSampler, uvY).xy * 2.0 - 1.0;
float2 tZ = lerp(Texture2DSample(BottomNormal, BottomNormalSampler, uvB).xy, Texture2DSample(TopNormal, TopNormalSampler, uvT).xy, up) * 2.0 - 1.0;
tX.y *= NormalY; tY.y *= NormalY; tZ.y *= NormalY;
// Each projection's tangent axes: X uses (Y, Z), Y uses (X, Z), Z uses (X, Y).
float3 nX = float3(sign(n.x) * sqrt(saturate(1.0 - dot(tX, tX))), tX.x, tX.y);
float3 nY = float3(tY.x, sign(n.y) * sqrt(saturate(1.0 - dot(tY, tY))), tY.y);
float3 nZ = float3(tZ.x, tZ.y, sign(n.z) * sqrt(saturate(1.0 - dot(tZ, tZ))));
Normal = normalize(nX * w.x + nY * w.y + nZ * w.z);
return albedo * Tint.rgb;
"""


def scan_master(defaults):
    """M_RubbleScan: three Megascans surfaces, triplanar in the instance's frame (SCAN_TRIPLANAR);
    `defaults` a scan's maps (each parameter's default must match its sampler type)."""
    m = material("M_RubbleScan")
    m.set_editor_property("tangent_space_normal", False)
    p, n = instance_space(m, -1800, 0)
    inputs = {"P": (p, ""), "N": (n, "")}
    y = 400
    for face in ("Top", "Side", "Bottom"):
        for role in ("Albedo", "Normal", "Roughness", "AO"):
            inputs[face + role] = (texture_param(m, face + role, defaults[role], SAMPLERS[role], -1800, y), "")
            y += 120
        inputs[face + "Size"] = (scalar(m, face + "Size", 2.0, -1800, y), "")
        y += 80
    inputs.update({
        "Tint": (vector(m, "Tint", (1, 1, 1, 1), -1800, y), ""),
        "RoughnessScale": (scalar(m, "RoughnessScale", 1.0, -1800, y + 80), ""),
        "Variation": (scalar(m, "Variation", 0.35, -1800, y + 160), ""),
        "NormalY": (scalar(m, "NormalY", 1.0, -1800, y + 240), ""),
    })
    c = custom(m, SCAN_TRIPLANAR, -1100, 0, inputs, {"Normal": 3, "Roughness": 1, "Occlusion": 1}, "ScanTriplanar")
    mel.connect_material_property(c, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(normal_to_world(m, c, -700, 150), "", MP.MP_NORMAL)
    mel.connect_material_property(c, "Roughness", MP.MP_ROUGHNESS)
    mel.connect_material_property(c, "Occlusion", MP.MP_AMBIENT_OCCLUSION)
    mel.connect_material_property(scalar(m, "Metallic", 0.0, -700, 400), "", MP.MP_METALLIC)
    mel.recompile_material(m)
    return m


def scan_instance(name, parent, top, side=None, bottom=None, scalars=None, vectors=None):
    mi = instance(name, parent, scalars, vectors)
    for face, surf in (("Top", top), ("Side", side or top), ("Bottom", bottom or top)):
        maps, size = surf
        for role, tex in maps.items():
            mel.set_material_instance_texture_parameter_value(mi, face + role, tex)
        mel.set_material_instance_scalar_parameter_value(mi, face + "Size", size)
    eal.save_loaded_asset(mi)
    return mi


# Painted or bare metal from a Megascans imperfection scan (Scratched Painted Metal: a scratch mask
# and roughness): paint over metal, bare where scratched; triplanar in the instance's frame.
SCAN_METAL = r"""
float3 n = normalize(N);
float3 w = pow(abs(n), 8.0); w /= (w.x + w.y + w.z);
float s = 1.0 / (Size * 100.0);
float mX = Texture2DSample(Mask, MaskSampler, P.yz * s).r, mY = Texture2DSample(Mask, MaskSampler, P.xz * s).r, mZ = Texture2DSample(Mask, MaskSampler, P.xy * s).r;
float rX = Texture2DSample(Rough, RoughSampler, P.yz * s).r, rY = Texture2DSample(Rough, RoughSampler, P.xz * s).r, rZ = Texture2DSample(Rough, RoughSampler, P.xy * s).r;
float scratch = saturate((mX * w.x + mY * w.y + mZ * w.z) * ScratchAmount);
float r = rX * w.x + rY * w.y + rZ * w.z;
Metal = lerp(PaintMetal, 1.0, scratch);
Roughness = saturate(lerp(PaintRoughness, BareRoughness, scratch) * lerp(0.7, 1.3, r));
return lerp(PaintColor.rgb, BareColor.rgb, scratch);
"""


def metal_master(mask, rough):
    m = material("M_RubbleScanMetal")
    p, n = instance_space(m, -1800, 0)
    inputs = {
        "P": (p, ""), "N": (n, ""),
        "Mask": (texture_param(m, "Mask", mask, SAMPLERS["Roughness"], -1800, 400), ""),
        "Rough": (texture_param(m, "Rough", rough, SAMPLERS["Roughness"], -1800, 520), ""),
        "Size": (scalar(m, "Size", 1.0, -1800, 640), ""),
        "PaintColor": (vector(m, "PaintColor", (0.05, 0.055, 0.06, 1), -1800, 720), ""),
        "BareColor": (vector(m, "BareColor", (0.56, 0.57, 0.58, 1), -1800, 800), ""),
        "PaintMetal": (scalar(m, "PaintMetal", 0.0, -1800, 880), ""),
        "PaintRoughness": (scalar(m, "PaintRoughness", 0.45, -1800, 960), ""),
        "BareRoughness": (scalar(m, "BareRoughness", 0.3, -1800, 1040), ""),
        "ScratchAmount": (scalar(m, "ScratchAmount", 1.0, -1800, 1120), ""),
    }
    c = custom(m, SCAN_METAL, -1100, 0, inputs, {"Metal": 1, "Roughness": 1}, "ScanMetal")
    mel.connect_material_property(c, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(c, "Metal", MP.MP_METALLIC)
    mel.connect_material_property(c, "Roughness", MP.MP_ROUGHNESS)
    mel.recompile_material(m)
    return m


# The ground: a Megascans surface in world space (XY), sampled at two scales and rotations and
# mixed by broad noise, so its 2 m scan does not read as tiles across a 400 m plane.
SCAN_GROUND = r"""
float2 uv = WorldP.xy / (Size * 100.0);
float2 uv2 = float2(uv.x * 0.8 - uv.y * 0.6, uv.x * 0.6 + uv.y * 0.8) * 0.43 + 0.37;
float k = saturate(0.5 + 0.5 * sin(WorldP.x / 2300.0) * sin(WorldP.y / 1900.0 + 1.3));
float3 a = lerp(Texture2DSample(Albedo, AlbedoSampler, uv).rgb, Texture2DSample(Albedo, AlbedoSampler, uv2).rgb, k);
float2 t = lerp(Texture2DSample(NormalMap, NormalMapSampler, uv).xy, Texture2DSample(NormalMap, NormalMapSampler, uv2).xy, k) * 2.0 - 1.0;
t.y *= NormalY;
Normal = float3(t, sqrt(saturate(1.0 - dot(t, t))));
Roughness = lerp(Texture2DSample(Rough, RoughSampler, uv).r, Texture2DSample(Rough, RoughSampler, uv2).r, k);
Occlusion = lerp(Texture2DSample(AO, AOSampler, uv).r, Texture2DSample(AO, AOSampler, uv2).r, k);
return a;
"""


def scan_ground(surf):
    maps, size = surf
    m = material("M_RubbleScanGround")
    m.set_editor_property("used_with_instanced_static_meshes", False)
    inputs = {
        "WorldP": (expr(m, unreal.MaterialExpressionWorldPosition, -1600, 0), ""),
        "Albedo": (texture_param(m, "Albedo", maps["Albedo"], SAMPLERS["Albedo"], -1600, 150), ""),
        "NormalMap": (texture_param(m, "NormalMap", maps["Normal"], SAMPLERS["Normal"], -1600, 270), ""),
        "Rough": (texture_param(m, "Rough", maps["Roughness"], SAMPLERS["Roughness"], -1600, 390), ""),
        "AO": (texture_param(m, "AO", maps["AO"], SAMPLERS["AO"], -1600, 510), ""),
        "Size": (scalar(m, "Size", size, -1600, 630), ""),
        "NormalY": (scalar(m, "NormalY", 1.0, -1600, 710), ""),
    }
    c = custom(m, SCAN_GROUND, -1000, 0, inputs, {"Normal": 3, "Roughness": 1, "Occlusion": 1}, "ScanGround")
    mel.connect_material_property(c, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(c, "Normal", MP.MP_NORMAL)
    mel.connect_material_property(c, "Roughness", MP.MP_ROUGHNESS)
    mel.connect_material_property(c, "Occlusion", MP.MP_AMBIENT_OCCLUSION)
    mel.recompile_material(m)
    return instance("MI_Scan_Ground", m)


def megascans():
    """Rubble's own looks from Quixel Megascans (MI_Scan_*): the glass tower's slabs, steel and
    aluminium, and the ground. A missing scan is reported with its Fab listing and its look skipped
    (the renderer falls back)."""
    if not eal.does_directory_exist(SCANS):
        eal.make_directory(SCANS)
    want = ["smooth_precast_concrete", "shuttered_concrete_wall", "stained_concrete_ceiling", "grassy_soil"]
    sets = {k: scan_set(k) for k in want}
    scan_missing([k for k, v in sets.items() if v is None])
    first = next((v for v in sets.values() if v), None)
    scan = scan_master(first[0]) if first else None
    if scan and sets["smooth_precast_concrete"] and sets["shuttered_concrete_wall"] and sets["stained_concrete_ceiling"]:
        # Floor slabs: precast tops, shuttered (board-formed) edges, stained ceilings below.
        scan_instance("MI_Scan_Slab", scan, sets["smooth_precast_concrete"], sets["shuttered_concrete_wall"], sets["stained_concrete_ceiling"])
    if scan and sets["shuttered_concrete_wall"]:
        scan_instance("MI_Scan_Concrete", scan, sets["shuttered_concrete_wall"])
    mask = import_scan("scratched_painted_metal", "Mask", "grey")
    rough = import_scan("scratched_painted_metal", "Roughness", "grey")
    if mask and rough:
        metal = metal_master(mask, rough)
        size = scan_size("scratched_painted_metal")
        # Steel columns and frames: charcoal paint, scratched through to steel.
        instance("MI_Scan_Steel", metal, {"Size": size, "ScratchAmount": 0.9}, {"PaintColor": (0.035, 0.038, 0.042, 1)})
        # Spandrel panels: clear-anodised aluminium, faintly scuffed.
        instance("MI_Scan_Aluminium", metal, {"Size": size, "PaintMetal": 1.0, "PaintRoughness": 0.32, "BareRoughness": 0.22, "ScratchAmount": 0.35},
                 {"PaintColor": (0.62, 0.64, 0.67, 1), "BareColor": (0.8, 0.81, 0.83, 1)})
    else:
        scan_missing(["scratched_painted_metal"])
    if sets["grassy_soil"]:
        scan_ground(sets["grassy_soil"])
    eal.save_directory(SCANS, only_if_is_dirty=True, recursive=True)


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
    soft_car()
    g = ground()
    instance("MI_GroundYard", g, {"Roughness": 0.85}, {"BaseColor": (0.06, 0.06, 0.065, 1)})
    instance("MI_GroundHall", g, {"Roughness": 0.7}, {"BaseColor": (0.32, 0.32, 0.31, 1)})
    instance("MI_GroundLot", g, {"Roughness": 0.9}, {"BaseColor": (0.09, 0.085, 0.08, 1)})
    dust()
    spark()
    nanite_meshes()
    town()
    megascans()
    eal.save_directory(ROOT, only_if_is_dirty=False, recursive=True)
    log("done: " + ", ".join(sorted(a.split("/")[-1].split(".")[0] for a in eal.list_assets(ROOT))))


main()
