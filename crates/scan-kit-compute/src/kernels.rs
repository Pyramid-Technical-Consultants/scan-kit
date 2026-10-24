//! Scientific kernels. The numeric checks live in `scan-kit-core`.
//! These shaders are the GPU form of the same operations.

const SPLAT: &str = r#"
struct Spot { x: f32, y: f32, z: f32, weight: f32, sigma: f32 }
struct Grid { origin: vec3<f32>, spacing: vec3<f32>, size: vec3<u32> }

@group(0) @binding(0) var<storage, read> spots: array<Spot>;
@group(0) @binding(1) var<storage, read_write> dose: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> grid: Grid;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&spots)) { return; }
    let spot = spots[id.x];
    let reach = u32(ceil(3.0 * spot.sigma / grid.spacing.x));
    let center = vec3<i32>(vec3<f32>(
        (spot.x - grid.origin.x) / grid.spacing.x,
        (spot.y - grid.origin.y) / grid.spacing.y,
        (spot.z - grid.origin.z) / grid.spacing.z
    ));
    let inv = 1.0 / (2.0 * spot.sigma * spot.sigma);
    for (var dz = 0u; dz <= reach * 2u; dz++) {
        for (var dy = 0u; dy <= reach * 2u; dy++) {
            for (var dx = 0u; dx <= reach * 2u; dx++) {
                let ix = center.x + i32(dx) - i32(reach);
                let iy = center.y + i32(dy) - i32(reach);
                let iz = center.z + i32(dz) - i32(reach);
                if (ix < 0 || iy < 0 || iz < 0) { continue; }
                if (u32(ix) >= grid.size.x || u32(iy) >= grid.size.y || u32(iz) >= grid.size.z) { continue; }
                let p = grid.origin + vec3<f32>(f32(ix), f32(iy), f32(iz)) * grid.spacing;
                let delta = p - vec3<f32>(spot.x, spot.y, spot.z);
                let value = spot.weight * exp(-dot(delta, delta) * inv);
                let index = u32(ix) + grid.size.x * (u32(iy) + grid.size.y * u32(iz));
                atomicAdd(&dose[index], u32(max(value, 0.0) * 1000.0));
            }
        }
    }
}
"#;

const RAYMARCH: &str = r#"
struct View { size: vec2<u32>, depth: u32 }

@group(0) @binding(0) var<storage, read> dose: array<f32>;
@group(0) @binding(1) var<storage, read_write> image: array<f32>;
@group(0) @binding(2) var<uniform> view: View;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.size.x || id.y >= view.size.y) { return; }
    var peak = 0.0;
    for (var z = 0u; z < view.depth; z++) {
        let sample = dose[id.x + view.size.x * (id.y + view.size.y * z)];
        peak = max(peak, sample);
    }
    image[id.x + view.size.x * id.y] = peak;
}
"#;

const GAMMA: &str = r#"
@group(0) @binding(0) var<storage, read> reference: array<f32>;
@group(0) @binding(1) var<storage, read> evaluated: array<f32>;
@group(0) @binding(2) var<storage, read_write> gamma: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&reference)) { return; }
    let dose_delta = (evaluated[i] - reference[i]) / max(reference[i], 1e-6);
    gamma[i] = abs(dose_delta);
}
"#;

const DVH: &str = r#"
@group(0) @binding(0) var<storage, read> dose: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&dose)) { return; }
    let bin = min(u32(max(dose[i], 0.0) * 10.0), arrayLength(&hist) - 1u);
    atomicAdd(&hist[bin], 1u);
}
"#;

const RESAMPLE: &str = r#"
struct Shape { src: vec3<u32>, dst: vec3<u32> }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> dest: array<f32>;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= shape.dst.x || id.y >= shape.dst.y || id.z >= shape.dst.z) { return; }
    let sx = min(id.x * shape.src.x / max(shape.dst.x, 1u), shape.src.x - 1u);
    let sy = min(id.y * shape.src.y / max(shape.dst.y, 1u), shape.src.y - 1u);
    let sz = min(id.z * shape.src.z / max(shape.dst.z, 1u), shape.src.z - 1u);
    let src_index = sx + shape.src.x * (sy + shape.src.y * sz);
    let dst_index = id.x + shape.dst.x * (id.y + shape.dst.y * id.z);
    dest[dst_index] = source[src_index];
}
"#;

const MC: &str = include_str!("mc_transport.wgsl");

pub fn compile_scientific_shaders() -> Result<(), String> {
    for source in [SPLAT, RAYMARCH, GAMMA, DVH, RESAMPLE] {
        compile(source)?;
    }
    // The transport shader expects the host to prepend `struct Params`.
    compile(MC)?;
    Ok(())
}

fn compile(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|err| err.to_string())?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_027_through_031_scientific_shaders_compile() {
        compile_scientific_shaders().unwrap();
    }
}
