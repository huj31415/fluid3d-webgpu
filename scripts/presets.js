let presetXOffset = 32;
let barrierThickness = 3;

const shapes = Object.freeze({
  circular: (y, z) => y * y + z * z,
  linear: (y, z) => y * y,
  square: (y, z) => Math.max(y * y, z * z)
});

const presetSettings = {
  DoubleSlit: { slitWidth: 8, slitSpacing: 32, slitHeight: 64 },
  Aperture: { shape: shapes.circular, invert: false },
  Prism: { n: 3, rot: (10).toRad(), width: 0.5 },
  FlatWing: { chord: 64, thickness: 3, AoA: (10).toRad(), width: 0.5 },
}

/**
 * Updates the barriers at a given coordinate
 * @param {Number} x X coordinate
 * @param {Number} yRel Y coordinate relative to y midpoint
 * @param {Number} zRel Z coordinate relative to z midpoint
 * @param {Number} newValue New barrier value (0=barrier,1=free) at the specified coordinate
 */
function updateQuadSymmetry(x, yRel, zRel, newValue) {
  if (x < simulationDomain[0] && x >= 0
    && Math.abs(yRel) <= yMidpt
    && Math.abs(zRel) <= zMidpt
  )
    [
      index3d(x, yMidpt - yRel, zMidpt - zRel),
      index3d(x, yMidpt - yRel, zMidpt + zRel),
      index3d(x, yMidpt + yRel, zMidpt - zRel),
      index3d(x, yMidpt + yRel, zMidpt + zRel)
    ].forEach(i => barrierData[i] = newValue);
}

/**
 * Writes the speed texture to the gpu
 * @param {Boolean} reset Whether to clear all barriers 
 */
function updateBarrierTexture(reset = false) {
  clearPressure = true;
  cleared = reset;
  device.queue.writeTexture(
    { texture: storage.barrierTex },
    reset ? barrierData.fill(1) : barrierData,
    { offset: 0, bytesPerRow: simulationDomain[0] * 1, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
}

const lerp = (value, in_min, in_max, out_min, out_max) => ((value - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min;
const mod = (x, a) => x - a * Math.floor(x / a);

const flatPresets = Object.freeze({
  DoubleSlit: (y, z, args = presetSettings.DoubleSlit) => (
    y > args.slitHeight / 2 // fill outside of slit area
      || (y <= args.slitHeight / 2 // fill if inside slit height and outside slit opening
        && (z < (args.slitSpacing - args.slitWidth) / 2 || z > (args.slitSpacing + args.slitWidth) / 2)
      ) ? 0 : 1
  ),
  // cutout grid / 2d version of double slit
  Aperture: (y, z, args = presetSettings.Aperture) => (args.shape(y, z) >= sharedSettings.radius * sharedSettings.radius) ? args.invert ? 1 : 0 : args.invert ? 0 : 1,
});

/**
 * Updates barrier texture to include a 4 way symmetric barrier
 * @param {Function} preset Boolean function -> true: add a barrier, false: no barrier
 * @param {Number} distance X distance from x=0
 * @param {Number} thickness Thickness of the barrier
 * @param {Object} args Object containing the arguments for the selected preset
 */
function quadSymmetricFlatPreset(preset, distance = 64, thickness = 2, args) {
  for (let z = 0; z <= zMidpt; z++) {
    for (let y = 0; y <= yMidpt; y++) {
      for (let x = distance; x < distance + thickness; x++) {
        updateQuadSymmetry(x, y, z, preset(y, z, args));
      }
    }
  }
  updateBarrierTexture();
}

function nGonPrism(distance = presetXOffset, args = presetSettings.Prism) {
  const sectorAngle = Math.PI / args.n;
  const num = Math.cos(sectorAngle) * sharedSettings.radius;
  const rot = args.rot;//.toRad();
  
  for (let z = 0; z < simulationDomain[2] * args.width; z++) {
    for (let y = -sharedSettings.radius; y < sharedSettings.radius; y++) {
      for (let x = -sharedSettings.radius; x < sharedSettings.radius; x++) {
        if (Math.hypot(y, -x) < num / Math.cos(mod(Math.atan2(y, -x) - rot, 2 * sectorAngle) - sectorAngle)) {
          barrierData[index3d(x + distance + sharedSettings.radius, y + yMidpt, z)] = 0;
        }
      }
    }
  }
  updateBarrierTexture();
}

function flatWing(distance = presetXOffset, args = presetSettings.FlatWing) {
  const AoA = args.AoA;//.toRad();
  const height = Math.ceil(args.chord * Math.sin(AoA)) + args.thickness;
  const length = Math.ceil(args.chord * Math.cos(AoA)) + args.thickness;
  const halfHeight = Math.ceil(height / 2);
  const halfLength = Math.ceil(length / 2);

  for (let z = 0; z < simulationDomain[2] * args.width; z++) {
    for (let y = -halfHeight; y < halfHeight; y++) {
      for (let x = -halfLength; x < halfLength; x++) {
        if (Math.abs(y - x * (-height / length)) <= args.thickness) {
          barrierData[index3d(x + distance + halfLength, y + yMidpt, z)] = 0;
        }
      }
    }
  }
  updateBarrierTexture();
}