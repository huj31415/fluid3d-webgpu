const initShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}

// @group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var velOld:  texture_storage_3d<rgba${readOrWriteFormat}float, write>;
@group(0) @binding(2) var velNew:  texture_storage_3d<rgba${readOrWriteFormat}float, write>;
@group(0) @binding(3) var smokeOld:  texture_storage_3d<rg${readOrWriteFormat}float, write>;
@group(0) @binding(4) var smokeNew:  texture_storage_3d<rg${readOrWriteFormat}float, write>;

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  // textureStore(pressure, gid, vec4f(0));
  // textureStore(velOld, gid, vec4f(uni.vInflow,0,0,0));
  // textureStore(velNew, gid, vec4f(uni.vInflow,0,0,0));
  textureStore(velOld, gid, vec4f(0));
  textureStore(velNew, gid, vec4f(0));

  let gid_f = vec3f(gid);
  let smoke = vec4f(0,1,0,0);

  textureStore(smokeOld, gid, smoke);
  textureStore(smokeNew, gid, smoke);
}
`;

const clearPressureRefreshSmokeShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var pressure:  texture_storage_3d<r${readOrWriteFormat}float, write>;
@group(0) @binding(2) var smokeOld:  texture_storage_3d<rg${readOrWriteFormat}float, write>;
@group(0) @binding(3) var smokeNew:  texture_storage_3d<rg${readOrWriteFormat}float, write>;

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  textureStore(pressure, gid, vec4f(0));

  let gid_f = vec3f(gid);

  // if (gid.x == 0 && all(abs(gid_f.yz - uni.smokePos.xy) <= uni.smokeHalfSize)) {
  if (gid.x == 0 && gid.y > 0 && gid.y < u32(uni.volSize.y) - 1) {
    let smoke = vec4f(
      f32(abs(gid_f.z - uni.smokePos.x) <= uni.smokeHalfSize.x && (gid_f.y + uni.smokePos.y) % (2 * uni.smokeHalfSize.y) < uni.smokeHalfSize.z * uni.smokeHalfSize.y),
      uni.smokeTemp,0,0
    );
    // smoke.x = f32();
    textureStore(smokeOld, gid, smoke);
    textureStore(smokeNew, gid, smoke);
  }
}
`;

// create barrier bitmask texture
// 1 if barrier, 0 if open, 8 bits: unused,self,-x,+x,-y,+y,-z,+z
// can use countOneBits to get number of barriers around cell if necessary
// check barriers with mask & (1 << directionIndex)
const barrierMaskShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}

// @group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var barrierTex:  texture_3d<f32>;
@group(0) @binding(2) var barrierMask: texture_storage_3d<r8uint, write>;

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

const directions: array<vec3i, 7> = array<vec3i, 7>(
  // 00-05 orthogonal directions (cubic faces)
  vec3i(-1,  0,  0), // xn
  vec3i( 1,  0,  0), // xp
  vec3i( 0, -1,  0), // yn
  vec3i( 0,  1,  0), // yp
  vec3i( 0,  0, -1), // zn
  vec3i( 0,  0,  1), // zp
  vec3i( 0,  0,  0)  // self
);

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let gid_i = vec3i(gid);
  var mask = 0u;
  for (var i = 0u; i < 7u; i += 1) {
    if (textureLoad(barrierTex, gid_i + directions[i], 0).x == 0) {
      mask |= (1u << i);
    }
  }
  textureStore(barrierMask, gid, vec4u(mask,0,0,0));
}
`;

// Advect velocity using semi-Lagrangian scheme (implement MacCormack later)
// Advect smoke and add force based on temperature and pressure
const advectionShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var velOld:     texture_3d<f32>;
@group(0) @binding(2) var velNew:     texture_storage_3d<rgba${readOrWriteFormat}float, write>;
@group(0) @binding(3) var barrierTex: texture_3d<f32>;
@group(0) @binding(4) var linSampler: sampler;
@group(0) @binding(5) var smokeOld:   texture_3d<f32>;
@group(0) @binding(6) var smokeNew:   texture_storage_3d<rg${readOrWriteFormat}float, write>;

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

// Velocity advection compute shader
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let gid_f = vec3f(gid);
  // check if the index is within bounds
  if (any(gid_f >= uni.volSize)) { return; }

  // don't advect into barriers
  if (textureLoad(barrierTex, gid, 0).r == 0) {
    textureStore(smokeNew, gid, vec4f(0,1,0,0)); // can modify to apply temperature to objects
    return;
  }

  var newVel = vec4f(uni.vInflow, 0, 0, 0);
  let pastPos = gid_f - uni.dt * textureLoad(velOld, gid, 0).xyz;
  let pastPosNorm = saturate((pastPos + vec3f(0.5)) / uni.volSize); // velocity is in voxels/sec
  var newSmoke = textureSampleLevel(smokeOld, linSampler, pastPosNorm, 0);

  if (gid.x > 1 && gid.x < u32(uni.volSize.x) - 1) {
    // reverse trace particle velocity
    newVel = textureSampleLevel(velOld, linSampler, pastPosNorm, 0);
  }

  // let pastPos_i = vec3i(clamp(round(pastPos), vec3f(1), vec3f(uni.volSize) - vec3f(2)));

  let temp = newSmoke.y;
  let ambientTemp = 1.0;
  // (
  //     select(textureLoad(smokeOld, pastPos_i + vec3i(-1,0,0), 0).y, 1, pastPos_i.x == 0)
  //   + select(textureLoad(smokeOld, pastPos_i + vec3i( 1,0,0), 0).y, 1, pastPos_i.x == i32(uni.volSize.x) - 1)
  //   + select(textureLoad(smokeOld, pastPos_i + vec3i(0,-1,0), 0).y, 1, pastPos_i.y == 0)
  //   + select(textureLoad(smokeOld, pastPos_i + vec3i(0, 1,0), 0).y, 1, pastPos_i.y == i32(uni.volSize.y) - 1)
  //   + select(textureLoad(smokeOld, pastPos_i + vec3i(0,0,-1), 0).y, 1, pastPos_i.z == 0)
  //   + select(textureLoad(smokeOld, pastPos_i + vec3i(0,0, 1), 0).y, 1, pastPos_i.z == i32(uni.volSize.z) - 1)
  // ) / 6.0;

  newVel += vec4f(0, (temp - ambientTemp), 0, 0); // rho1 = mP/(1 + (t1 - t0)), F=(rho_surrounding - rho)*g*V
  if (gid.x > 0) {
    newSmoke.y = newSmoke.y - (temp - ambientTemp) * 0.001 * uni.dt; // equalize smoke temp with surroundings
  }

  // interpolate and advect velocity
  textureStore(velNew, gid, newVel);
  textureStore(smokeNew, gid, newSmoke);
}
`;

// Compute divergence and curl of velocity field
const velDivShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var vel:  texture_3d<f32>;
@group(0) @binding(2) var div:  texture_storage_3d<r${readOrWriteFormat}float, write>;
@group(0) @binding(3) var curl: texture_storage_3d<rgba${readOrWriteFormat}float, write>;
@group(0) @binding(4) var barrierMaskTex: texture_3d<u32>;

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

const directions: array<vec3i, 6> = array<vec3i, 6>(
  // 00-05 orthogonal directions (cubic faces)
  vec3i(-1,  0,  0), // xn
  vec3i( 1,  0,  0), // xp
  vec3i( 0, -1,  0), // yn
  vec3i( 0,  1,  0), // yp
  vec3i( 0,  0, -1), // zn
  vec3i( 0,  0,  1), // zp
);

// Velocity divergence compute shader
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let gid_i = vec3i(gid);

  // check if the index is within bounds
  if (any(gid >= vec3u(uni.volSize))) { return; }

  let barrierMask = textureLoad(barrierMaskTex, gid_i, 0).r;
  if ((barrierMask & (1u << 6)) == 1) { return; }

  var divV = 0.0;

  // curl = (dvz/dy - dvy/dz, dvx/dz - dvz/dx, dvy/dx - dvx/dy)
  var curlV = vec3f(0);
  
  // only consider non-barrier (barrierMask == 0) neighbors
  if ((barrierMask & (1u << 0)) == 0) {
    let vel = textureLoad(vel, gid_i + directions[0], 0).xyz;
    divV -= vel.x;
    curlV += vec3f(0, vel.z, -vel.y);
  }
  if ((barrierMask & (1u << 1)) == 0) {
    let vel = textureLoad(vel, gid_i + directions[1], 0).xyz;
    divV += vel.x;
    curlV += vec3f(0, -vel.z, vel.y);
  }
  if ((barrierMask & (1u << 2)) == 0) {
    let vel = textureLoad(vel, gid_i + directions[2], 0).xyz;
    divV -= vel.y;
    curlV += vec3f(-vel.z, 0, vel.x);
  }
  if ((barrierMask & (1u << 3)) == 0) {
    let vel = textureLoad(vel, gid_i + directions[3], 0).xyz;
    divV += vel.y;
    curlV += vec3f(vel.z, 0, -vel.x);
  }
  if ((barrierMask & (1u << 4)) == 0) {
    let vel = textureLoad(vel, gid_i + directions[4], 0).xyz;
    divV -= vel.z;
    curlV += vec3f(vel.y, -vel.x, 0);
  }
  if ((barrierMask & (1u << 5)) == 0) {
    let vel = textureLoad(vel, gid_i + directions[5], 0).xyz;
    divV += vel.z;
    curlV += vec3f(-vel.y, vel.x, 0);
  }

  // store divergence with dx = 1
  textureStore(div, gid, vec4f(divV * 0.5, 0, 0, 0));
  // store curl with dx = 1
  if (uni.visMode == 5 && all(gid < vec3u(uni.volSize) - vec3u(1)) && all(gid > vec3u(0))) {
    textureStore(curl, gid, vec4f(curlV * 0.5, 0));
  }
}
`;

// Calculate pressure using pressure Poisson equation and red-black Gauss-Seidel iteration
// Periodically zero pressure field after 50-100 frames or after significant updates
// Merge with divergence shader?
const pressureShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}
@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var velDiv:   texture_storage_3d<r${readOrWriteFormat}float, read>;
@group(0) @binding(2) var pressure: texture_storage_3d<r${readAndWriteFormat}float, read_write>;
@group(0) @binding(3) var barrierMaskTex:  texture_3d<u32>;

const directions: array<vec3i, 6> = array<vec3i, 6>(
  // 00-05 orthogonal directions (cubic faces)
  vec3i(-1,  0,  0), // xn
  vec3i( 1,  0,  0), // xp
  vec3i( 0, -1,  0), // yn
  vec3i( 0,  1,  0), // yp
  vec3i( 0,  0, -1), // zn
  vec3i( 0,  0,  1), // zp
);

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

// 2 wide halo may have better stability
var<workgroup> tile: array<f32, (WG_X + 2) * (WG_Y + 2) * (WG_Z + 2)>;

fn tileIndex(idx: vec3i) -> u32 {
  let sidx = vec3u(idx + vec3i(1)); // shift by 1 for halo
  return sidx.x + (WG_X + 2u) * (sidx.y + (WG_Y + 2u) * sidx.z);
}

fn neighborSum(gid: vec3u, currentPressure: f32, barrierMask: u32, indices: array<u32, 6>) -> f32 {
  // let pressureXn = select(currentPressure, tile[indices[0]], (barrierMask & (1 << 0)) == 0);
  let pressureXn = select(currentPressure, select(tile[indices[0]], 0, gid.x == u32(uni.volSize.x) - 1), (barrierMask & (1 << 0)) == 0);
  // let pressureXp = select(currentPressure, select(tile[indices[1]], 0, gid.x == u32(uni.volSize.x) - 1), (barrierMask & (1 << 1)) == 0);
  let pressureXp = select(currentPressure, tile[indices[1]], (barrierMask & (1 << 1)) == 0);
  let pressureYn = select(currentPressure, tile[indices[2]], (barrierMask & (1 << 2)) == 0);
  let pressureYp = select(currentPressure, tile[indices[3]], (barrierMask & (1 << 3)) == 0);
  let pressureZn = select(currentPressure, tile[indices[4]], (barrierMask & (1 << 4)) == 0);
  let pressureZp = select(currentPressure, tile[indices[5]], (barrierMask & (1 << 5)) == 0);

  return pressureXp + pressureXn + pressureYp + pressureYn + pressureZp + pressureZn;
}

// Pressure compute shader
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u
) {
  let gid_i = vec3i(gid);
  let lid_i = vec3i(lid);

  // textureStore(pressure, gid, vec4f(0));
  let div = textureLoad(velDiv, gid).r;

  let stride = vec3u(1, WG_X + 2, (WG_X + 2) * (WG_Y + 2));
  let currentTileIndex = tileIndex(lid_i);

  let indices = array<u32, 6>(
    currentTileIndex - stride.x,
    currentTileIndex + stride.x,
    currentTileIndex - stride.y,
    currentTileIndex + stride.y,
    currentTileIndex - stride.z,
    currentTileIndex + stride.z
  );

  let pressureValue = textureLoad(pressure, gid_i).r;  
  let barrierMask = textureLoad(barrierMaskTex, gid, 0).r;
  
  let SORa = 1.0 - uni.SORomega;
  let SORb = uni.SORomega / 6.0;
  let rhs = textureLoad(velDiv, gid).r / uni.dt;

  let isRed = ((gid.x + gid.y + gid.z) & 1u) == 0u;

  // load into shared memory
  tile[currentTileIndex] = pressureValue;

  for (var d = 0; d < 6; d = d + 1) {
    let dir = directions[d];
    let haloIdx = tileIndex(lid_i + dir);
    let g = gid_i + dir; // global neighbor index
    if (all(g >= vec3i(0)) && all(g < vec3i(uni.volSize))) {
      tile[haloIdx] = textureLoad(pressure, g).r;
    } else {
      tile[haloIdx] = 0.0; // boundary condition (or mirror)
    }
  }
  workgroupBarrier();

  // red-black Gauss-Seidel iteration with overrelaxation
  for (var i = 0; i < i32(uni.pressureLocalIter); i = i + 1) {
    if (isRed) {
      let sum = neighborSum(gid, pressureValue, barrierMask, indices);
      tile[currentTileIndex] = SORa * tile[currentTileIndex] + SORb * (sum - rhs);
    }
    workgroupBarrier();
    if (!isRed) {
      let sum = neighborSum(gid, pressureValue, barrierMask, indices);
      tile[currentTileIndex] = SORa * tile[currentTileIndex] + SORb * (sum - rhs);
    }
    workgroupBarrier();
  }
  textureStore(pressure, gid, vec4f(tile[currentTileIndex], 0.0, 0.0, 0.0));
}
`;

// Compute gradient of pressure field, then subtract from velocity
const projectionShaderCode = (readOrWriteFormat = 16, readAndWriteFormat = 16) => /* wgsl */`
${uni.uniformStruct}
@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var velOld:   texture_storage_3d<rgba${readOrWriteFormat}float, read>;
@group(0) @binding(2) var velNew:   texture_storage_3d<rgba${readOrWriteFormat}float, write>;
@group(0) @binding(3) var pressure: texture_storage_3d<r${readOrWriteFormat}float, read>;
@group(0) @binding(4) var barrierMaskTex:  texture_3d<u32>;

const directions: array<vec3i, 6> = array<vec3i, 6>(
  // 00-05 orthogonal directions (cubic faces)
  vec3i(-1,  0,  0), // xn
  vec3i( 1,  0,  0), // xp
  vec3i( 0, -1,  0), // yn
  vec3i( 0,  1,  0), // yp
  vec3i( 0,  0, -1), // zn
  vec3i( 0,  0,  1), // zp
);

override WG_X: u32;
override WG_Y: u32;
override WG_Z: u32;

// 2 wide halo may have better stability
var<workgroup> tile: array<f32, (WG_X + 2) * (WG_Y + 2) * (WG_Z + 2)>;

fn tileIndex(idx: vec3i) -> u32 {
  let sidx = vec3u(idx + vec3i(1)); // shift by 1 for halo
  return sidx.x + (WG_X + 2u) * (sidx.y + (WG_Y + 2u) * sidx.z);
}

// Pressure projection compute shader
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u
) {

  let barrierMask = textureLoad(barrierMaskTex, gid, 0).r;
  if ((barrierMask & (1u << 6)) == 1) { return; }

  let gid_i = vec3i(gid);
  let lid_i = vec3i(lid);

  let pressureValue = textureLoad(pressure, gid_i).r;
  var oldVel = textureLoad(velOld, gid_i).xyz;

  // true if both sides open, otherwise false
  let vMask = vec3u((barrierMask & 3u), (barrierMask & (3u << 2)), (barrierMask & (3u << 4))) == vec3u(0);

  let pressureGrad = 0.5 * vec3f(
    select(pressureValue, textureLoad(pressure, gid_i + directions[1]).r, (barrierMask & (1u << 1)) == 0) -
    select(pressureValue, textureLoad(pressure, gid_i + directions[0]).r, (barrierMask & (1u << 0)) == 0),
    select(pressureValue, textureLoad(pressure, gid_i + directions[3]).r, (barrierMask & (1u << 3)) == 0) -
    select(pressureValue, textureLoad(pressure, gid_i + directions[2]).r, (barrierMask & (1u << 2)) == 0),
    select(pressureValue, textureLoad(pressure, gid_i + directions[5]).r, (barrierMask & (1u << 5)) == 0) -
    select(pressureValue, textureLoad(pressure, gid_i + directions[4]).r, (barrierMask & (1u << 4)) == 0)
  );

  if (gid.x < u32(uni.volSize.x - 1)) { oldVel -= pressureGrad; }

  let newVel = select(oldVel, vec3f(uni.vInflow, 0, 0), gid.x == 0) * vec3f(vMask);

  textureStore(velNew, gid, vec4f(newVel, 0.0));
}
`;

// maybe implement simple lighting by checking if barrier is between light source and sample point
const renderShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var stateTexture: texture_3d<f32>;
@group(0) @binding(2) var barrierTexture: texture_3d<f32>;
@group(0) @binding(3) var linSampler: sampler;
@group(0) @binding(4) var stateSampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) fragCoord: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
  );
  var output: VertexOut;
  output.position = vec4f(pos[vIdx], 0.0, 1.0);
  output.fragCoord = 0.5 * (pos[vIdx] + vec2f(1.0)) * uni.resolution;
  return output;
}

// // value to color: cyan -> blue -> transparent (0) -> red -> yellow
// fn transferFn(value: f32) -> vec4f {
//   // let a = 1.0 - pow(1.0 - clamp(value * value * 0.1, 0, 0.01), uni.rayDtMult);
//   let a = clamp(value * value * 0.1, 0, 0.01) * uni.globalAlpha;
//   return clamp(vec4f(value, (abs(value) - 1) * 0.5, -value, a), vec4f(0), vec4f(1)) * 10; // 10x for beer-lambert
// }

fn rayBoxIntersect(start: vec3f, dir: vec3f) -> vec2f {
  let box_min = vec3f(0);
  let box_max = uni.volSizeNorm;
  let inv_dir = 1.0 / dir;
  let tmin_tmp = (box_min - start) * inv_dir;
  let tmax_tmp = (box_max - start) * inv_dir;
  let tmin = min(tmin_tmp, tmax_tmp);
  let tmax = max(tmin_tmp, tmax_tmp);
  let t0 = max(tmin.x, max(tmin.y, tmin.z));
  let t1 = min(tmax.x, min(tmax.y, tmax.z));
  return vec2f(t0, t1);
}

fn pcgHash(input: f32) -> f32 {
  let state = u32(input) * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return f32((word >> 22) ^ word) / 4294967295.0;
}

// Convert linear color to sRGB
fn linear2srgb(color: vec4f) -> vec4f {
  let cutoff = color.rgb < vec3f(0.0031308);
  let higher = 1.055 * pow(color.rgb, vec3f(1.0 / 2.4)) - 0.055;
  let lower = 12.92 * color.rgb;
  return vec4f(select(higher, lower, cutoff), color.a);
}

@fragment
fn fs(@location(0) fragCoord: vec2f) -> @location(0) vec4f {
  // Convert fragment coordinates to normalized device coordinates
  let fragNdc = fragCoord / uni.resolution * 2.0 - 1.0;

  // Project NDC to world space
  let near = uni.invMatrix * vec4f(fragNdc, 0.0, 1.0);
  let far  = uni.invMatrix * vec4f(fragNdc, 1.0, 1.0);

  // ray origin and direction
  let rayOrigin = uni.cameraPos;
  let rayDir = normalize((far.xyz / far.w) - (near.xyz / near.w));

  let intersection = rayBoxIntersect(rayOrigin, rayDir);

  // discard if ray does not intersect the box
  if (intersection.x > intersection.y || intersection.y <= 0.0) {
    // discard;
    return vec4f(0.1);
  }

  let t0 = max(intersection.x, 0.0);

  let rayDtVec = 1.0 / (uni.volSize * abs(rayDir));
  let rayDt = uni.rayDtMult * min(rayDtVec.x, min(rayDtVec.y, rayDtVec.z));

  let offset = pcgHash(fragCoord.x + uni.resolution.x * fragCoord.y) * rayDt;
  var rayPos = (rayOrigin + (t0 + offset) * rayDir) / uni.volSizeNorm;
  let rayDirNorm = rayDir / uni.volSizeNorm;

  var color = vec4f(0);

  var remainingDist = intersection.y - t0;

  loop {
    if (remainingDist <= 0.0) { break; }

    let adjDt = min(rayDt, remainingDist);
    let samplePos = rayPos;

    // Precompute position increment
    rayPos += rayDirNorm * adjDt;
    remainingDist -= adjDt;

    // Sample barrier texture
    let barrier = textureSampleLevel(barrierTexture, linSampler, samplePos, 0).r;

    // Early exit on barrier
    if (barrier == 0.0 && (u32(uni.options) & 1u) == 1u) {
      // Barrier blend
      color += vec4f((1.0 - color.a) * (1.0 - exp(-adjDt))); // Barrier color
      break;
    }

    var sampleColor = vec4f(0);
    if (uni.visMode <= 2.0) { // 0: scalar-abs-bw, 1: scalar-color
      let sampleValue = uni.visMult * select(textureSampleLevel(stateTexture, stateSampler, samplePos, 0).x, textureSampleLevel(stateTexture, stateSampler, samplePos, 0).y - 1, uni.visMode == 2);//y-uni.smokeTemp; // scalar, also add option for y-1 for smoke temperature
      // Skip if empty and not a boundary
      if (sampleValue == 0.0 && barrier == 1.0) {
        continue;
      }
      let a = clamp(abs(sampleValue) * 0.01, 0, 0.01) * uni.globalAlpha;
      if (uni.visMode == 0.0) {
        sampleColor = 10 * saturate(abs(vec4f(sampleValue, sampleValue, sampleValue, a)));
      } else {
        sampleColor = 10 * saturate(vec4f(sampleValue, (sampleValue - 1) * 0.5, -sampleValue, a));
      }
    } else { // 3: vel-xyz-color, 4: vel-mag-color, 5: curl-xyz-color
      // Sample state
      let sampleValue = uni.visMult * (textureSampleLevel(stateTexture, stateSampler, samplePos, 0).xyz - select(vec3f(0), vec3f(uni.vInflow / 2, 0, 0), uni.visMode < 5.0)); // free velocity half of vInflow?
      // Skip if empty and not a boundary
      if (all(vec3f(sampleValue) == vec3f(0.0)) && barrier == 1.0) {
        continue;
      }
      // transfer function
      let c = select(abs(sampleValue), vec3f(length(sampleValue)), uni.visMode == 3.0);
      sampleColor = 10 * saturate(vec4f(c, clamp(length(sampleValue) * 0.01, 0, 0.01) * uni.globalAlpha));
    }

    // Exponential blending
    color += (1.0 - color.a) * (1.0 - exp(-sampleColor.a * adjDt)) * vec4f(sampleColor.xyz, 1);

    // Exit if nearly opaque
    // if (color.a >= 0.95) {
    //   break;
    // }
  }
  return linear2srgb(color);
}
`;