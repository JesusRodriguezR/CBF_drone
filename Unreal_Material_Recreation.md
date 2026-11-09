# Unreal Engine Material Recreation

The FBX exports preserve geometry and material slots, but Blender procedural node graphs do not transfer directly. Recreate the materials in Unreal Engine using the following profiles.

## Leaf material: `M_Pimiento_Leaf`

- Material Domain: `Surface`
- Blend Mode: `Masked` or `Opaque`
- Shading Model: `Two Sided Foliage`
- Two Sided: enabled
- Base Color: dark green
- Roughness: `0.65–0.8`
- Subsurface Color: yellow-green
- Subsurface amount: approximately `0.03–0.08`

For color variation:

```text
TextureCoordinate → Noise → Lerp(dark green, light green) → Base Color
```

Use another Noise node through a Normal/Bump function for subtle leaf surface variation.

## Stem material: `M_Pimiento_Stem`

- Shading Model: `Default Lit`
- Base Color: dark green/brown
- Roughness: `0.8–0.9`
- Add subtle normal noise
- Metallic: `0`

## Pepper fruit materials

Create separate materials for green, red, and yellow peppers.

- Shading Model: `Default Lit`
- Roughness: `0.3–0.45`
- Slight glossy response
- Very subtle subsurface value
- Use a Noise node for small color variation

## Dirt: `M_Greenhouse_Dirt`

- Base Color: brown Noise → ColorRamp
- Roughness: `0.9–1.0`
- Noise → Normal input
- Add subtle darker patches and fine surface variation

## Greenhouse polyethylene film: `M_Greenhouse_Film`

- Blend Mode: `Translucent`
- Two Sided: enabled
- Base Color: slightly off-white
- Roughness: `0.55–0.7`
- Opacity: approximately `0.15` for about 85% transmission
- Disable refraction initially
- Use `Surface Forward Shading` for translucent lighting

## Optional 50% shade cloth

Use a separate material and collection for the shade layer:

- Opacity: approximately `0.5`
- Prefer a woven/masked grid texture instead of solid translucent plastic

## FBX import workflow

Import these files:

- `UE_Exports/high_poly_pimiento_plant.fbx` — reusable UE plant asset
- `UE_Exports/high_poly_pimiento_variants_5.fbx` — plant variation set
- `UE_Exports/high_poly_pimiento_rows.fbx` — complete crop-row layout

Assign the Unreal materials to the imported material slots. The FBX preserves the slots, but Blender’s procedural node graphs do not transfer directly.

Import with scale `1.0`, then verify that the complete crop layout measures approximately 20 m along each row.
