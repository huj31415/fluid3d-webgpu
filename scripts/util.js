const uni = new Uniforms();
uni.addUniform("invMatrix", "mat4x4f");   // inverse proj*view matrix

uni.addUniform("cameraPos", "vec3f");     // camera position in world space
uni.addUniform("dt", "f32");              // simulation time step

uni.addUniform("volSize", "vec3f");       // volume size in voxels
uni.addUniform("rayDtMult", "f32");       // raymarch sampling factor

uni.addUniform("volSizeNorm", "vec3f");   // normalized volume size (volSize / max(volSize))
uni.addUniform("vInflow", "f32");         // inflow velocity

uni.addUniform("resolution", "vec2f");    // canvas resolution: x-width, y-height
uni.addUniform("pressureLocalIter", "f32"); // number of rbgs subiterations using local memory 
uni.addUniform("visMode", "f32");         // Visualization mode

uni.addUniform("smokeHalfSize", "vec3f"); // Smoke source size in Y and Z
uni.addUniform("SORomega", "f32");        // successive overrelaxation omega

uni.addUniform("smokePos", "vec2f");      // Smoke source center position
uni.addUniform("globalAlpha", "f32");     // global alpha multiplier
uni.addUniform("smokeTemp", "f32");       // smoke temperature

uni.addUniform("visMult", "f32");         // Field visualization multiplier
uni.addUniform("options", "f32");         // u32 bit-packed options - bit 0: barrier rendering on/off, bit 1: isosurface rendering on/off, bit 2: lighting on/off
uni.addUniform("isoMin", "f32");          // isosurface value
uni.addUniform("isoMax", "f32");          // isosurface value

uni.addUniform("lightDir", "vec3f");      // normalized directional light direction in world space
uni.addUniform("ambientIntensity", "f32");// ambient light intensity

uni.addUniform("lightColor", "vec3f");    // ambient light color * intensity

// visMode, options, pressureLocalIter can be packed
uni.finalize();

const storage = {
  velTex0: null,
  velTex1: null,
  divTex: null,
  pressureTex: null,
  smokeTemp0: null,
  smokeTemp1: null,
  curlTex: null,
  barrierTex: null,
};

let initialize = true, clearPressureRefreshSmoke = true, refreshBarriers = true, updateBarrierMask = true;

let renderTextureIdx = 4, pingPong = true;

let dt = 1;
let oldDt;

let options = 1;

let advectionMode = 0; // 0: MacCormack, 1: Semi-Lagrangian

let pressureGlobalIter = pressureGlobalIterTemp = 4;

const sharedSettings = {
  radius: 24,
  thickness: 16,
};

// simulation domain size [x, y, z], ex. [384, 256, 256], [512, 256, 384]
const simulationDomain = [384, 192, 192]; //[384, 256, 256];//[768, 384, 384];
let newDomainSize = vec3.clone(simulationDomain);
let simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];

let yMidpt = Math.floor(simulationDomain[1] / 2);
let zMidpt = Math.floor(simulationDomain[2] / 2);

const simulationDomainNorm = simulationDomain.map(v => v / Math.max(...simulationDomain));
let barrierData = new Uint8Array(simulationDomain[0] * simulationDomain[1] * simulationDomain[2]).fill(255);

let cleared = false;

let globalAlpha = 1;

let smokePos = [yMidpt, zMidpt];
const smokeHalfSize = [16, 8, 1];


function sphericalToCartesian(azimuth, elevation, distance) {
  const x = Math.cos(elevation) * Math.sin(azimuth);
  const y = Math.sin(elevation);
  const z = Math.cos(elevation) * Math.cos(azimuth);
  return vec3.scale([x, y, z], distance);
}
let lightDir = vec3.normalize(sphericalToCartesian(toRad(45), toRad(45), 1));
let lightIntensity = 5;
let lightColor = vec3.fromValues(1, 1, 1);
let ambientIntensity = 1;

/**
 * Resizes the simulation domain
 * @param {Array<Number>} newSize New simulation domain size
 */
function resizeDomain(newSize) {
  vec3.clone(newSize, simulationDomain);
  vec3.clone(simulationDomain.map(v => v / Math.max(...simulationDomain)), simulationDomainNorm);
  simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];
  barrierData = new Uint8Array(simVoxelCount).fill(255);
  yMidpt = Math.floor(simulationDomain[1] / 2);
  zMidpt = Math.floor(simulationDomain[2] / 2);
  camera.target = defaults.target = vec3.scale(simulationDomainNorm, 0.5);
  smokePos = [yMidpt, zMidpt];
  uni.values.smokePos.set(smokePos);
}

/**
 * Refreshes the active preset
 */
function refreshPreset(clear = false) {
  if (clear) barrierData.fill(255);
  clearPressureRefreshSmoke = refreshBarriers = true;
  const presetType = gui.io.presetSelect.value;
  switch (presetType) {
    case "DoubleSlit":
    case "Aperture":
      quadSymmetricFlatPreset(flatPresets[presetType], presetXOffset, barrierThickness, presetSettings[presetType]);
      break;
    case "Prism":
      nGonPrism();
      break;
    case "FlatWing":
      flatWing();
      break;
  }
}

function softReset() {
  initialize = true;
  clearPressureRefreshSmoke = true;
}

function hardReset() {
  initialize = true;
  clearPressureRefreshSmoke = true;
  pressureGlobalIter = pressureGlobalIterTemp;
  cancelAnimationFrame(rafId);
  clearInterval(perfIntId);
  if (!vec3.equals(simulationDomain, newDomainSize)) resizeDomain(newDomainSize);
  if (storage.barrierTex) storage.barrierTex.destroy();
  if (cleared) main();
  else main().then(refreshPreset);
}


const canvas = document.getElementById("canvas");

const gui = new GUI("3D fluid sim on WebGPU", canvas);

// Performance section
{
  gui.addGroup("perf", "Performance");
  gui.addStringOutput("res", "Resolution", "", "perf");
  gui.addHalfWidthGroups("perfL", "perfR", "perf");

  gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
  gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
  gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
  gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfL");
  gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfL");

  gui.addNumericOutput("vDivTime", "Div+Curl", "ms", 2, "perfR");
  gui.addNumericOutput("pressureTime", "Pressure", "ms", 2, "perfR");
  gui.addNumericOutput("vProjTime", "V Proj", "ms", 2, "perfR");
  gui.addNumericOutput("advectionTime", "Advection", "ms", 2, "perfR");
}

// Camera state section
{
  gui.addGroup("camState", "Camera state");
  gui.addNumericOutput("camFOV", "FOV", "°", 2, "camState");
  gui.addNumericOutput("camDist", "Dst", "", 2, "camState");
  gui.addStringOutput("camTarget", "Tgt", "", "camState");
  gui.addStringOutput("camPos", "Pos", "", "camState");
  gui.addNDimensionalOutput(["camAlt", "camAz"], "Alt/az", "°", ", ", 2, "camState");
}

// Sim controls
{
  gui.addGroup("simCtrl", "Sim controls");
  gui.addDropdown("visType", "Visualization", ["Smoke", "Velocity", "Velocity magnitude", "Pressure", "Temperature", "Curl", "Divergence"], "simCtrl", null, (value) => {
    pingPong = false;
    switch (value) {
      case "Smoke":
        renderTextureIdx = 4;
        pingPong = true;
        uni.values.visMode.set([0]);
        break;
      case "Velocity":
        renderTextureIdx = 0;
        uni.values.visMode.set([4]);
        break;
      case "Velocity magnitude":
        renderTextureIdx = 0;
        uni.values.visMode.set([3]);
        break;
      case "Pressure":
        renderTextureIdx = 3;
        uni.values.visMode.set([1]);
        break;
      case "Temperature":
        renderTextureIdx = 4;
        pingPong = true;
        uni.values.visMode.set([2]);
        break;
      case "Curl":
        renderTextureIdx = 6;
        uni.values.visMode.set([5]);
        break;
      case "Divergence":
        renderTextureIdx = 2;
        uni.values.visMode.set([1]);
        break;
    }
  });
  gui.addNumericInput("dt", true, "dt", { min: 0.5, max: 2, step: 0.01, val: dt, float: 2 }, "simCtrl", (newDt) => {
    if (oldDt) oldDt = newDt;
    else dt = newDt;
    uni.values.dt.set([dt]);
  }, "Simulation delta-time");
  gui.addNumericInput("inflowV", true, "Flow velocity", { min: 0, max: 5, step: 0.01, val: 2, float: 2 }, "simCtrl", (value) => uni.values.vInflow.set([value]));
  gui.addNumericInput("xSize", false, "X size (reinit)", { min: 8, max: 1024, step: 8, val: simulationDomain[0], float: 0 }, "simCtrl", (value) => newDomainSize[0] = value, "Requires reinitialization to apply");
  gui.addNumericInput("ySize", false, "Y size (reinit)", { min: 8, max: 512, step: 8, val: simulationDomain[1], float: 0 }, "simCtrl", (value) => newDomainSize[1] = value, "Requires reinitialization to apply");
  gui.addNumericInput("zSize", false, "Z size (reinit)", { min: 8, max: 512, step: 8, val: simulationDomain[2], float: 0 }, "simCtrl", (value) => newDomainSize[2] = value, "Requires reinitialization to apply");
  gui.addDropdown("advectionType", "Advection algorithm", ["MacCormack (2nd order)", "Semi-Lagrangian (1st order)"], "simCtrl", null, (value) => {
    switch (value) {
      case "MacCormack (2nd order)":
        advectionMode = 0;
        break;
      case "Semi-Lagrangian (1st order)":
        advectionMode = 1;
        break;
    }
  });
  gui.addNumericInput("pressureGlobalIter", true, "Press. global iter", { min: 2, max: 16, step: 1, val: pressureGlobalIter, float: 0 }, "simCtrl", (value) => pressureGlobalIterTemp = value, "Global pressure solver iterations per frame");
  gui.addNumericInput("pressureLocalIter", true, "Press. local iter", { min: 1, max: 16, step: 1, val: 4, float: 0 }, "simCtrl", (value) => uni.values.pressureLocalIter.set([value]), "Local pressure solver iterations per global iter.");
  gui.addButton("toggleSim", "Play / Pause", false, "simCtrl", () => {
    if (oldDt) {
      dt = oldDt;
      oldDt = null;
      uni.values.dt.set([dt]);
    } else {
      oldDt = dt;
      dt = 0;
    }
  });

  gui.addButton("softRestart", "Restart", false, "simCtrl", softReset);
  gui.addButton("hardRestart", "Reinitialize", true, "simCtrl", hardReset);
}

{
  gui.addGroup("smokeCtrl", "Smoke settings");
    gui.addNumericInput("smokePZ", true, "Z pos", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "smokeCtrl", (value) => {
    smokePos[0] = Math.round(value * simulationDomain[1]);
    uni.values.smokePos.set(smokePos);
    clearPressureRefreshSmoke = true;
  }, "Smoke source Y-coordinate normalized to simulation YZ plane");
  gui.addNumericInput("smokePY", true, "Y pos", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "smokeCtrl", (value) => {
    smokePos[1] = Math.round(value * simulationDomain[2]);
    uni.values.smokePos.set(smokePos);
    clearPressureRefreshSmoke = true;
  }, "Smoke source Z-coordinate normalized to simulation YZ plane");
  gui.addNumericInput("smokeSY", true, "Y size", { min: 0, max: 1, step: 0.01, val: smokeHalfSize[0] * 2 / simulationDomain[1], float: 2 }, "smokeCtrl", (value) => {
    smokeHalfSize[0] = Math.ceil(value * simulationDomain[1] / 2);
    uni.values.smokeHalfSize.set(smokeHalfSize);
    clearPressureRefreshSmoke = true;
  }, "Smoke source Y size normalized to simulation YZ plane");
  gui.addNumericInput("smokeSZ", true, "Z spacing", { min: 0, max: 1, step: 0.01, val: smokeHalfSize[1] * 2 / simulationDomain[2], float: 2 }, "smokeCtrl", (value) => {
    smokeHalfSize[1] = Math.ceil(value * simulationDomain[2] / 2);
    uni.values.smokeHalfSize.set(smokeHalfSize);
    clearPressureRefreshSmoke = true;
  }, "Smoke source Z spacing normalized to simulation YZ plane");
  gui.addNumericInput("smokeSZ", true, "Z size", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "smokeCtrl", (value) => {
    smokeHalfSize[2] = 2 * value;
    uni.values.smokeHalfSize.set(smokeHalfSize);
    clearPressureRefreshSmoke = true;
  }, "Smoke source Z size normalized to Z spacing");
  gui.addNumericInput("smokeTemp", true, "Temperature", { min: 0, max: 2, step: 0.01, val: 1, float: 2 }, "smokeCtrl", (value) => {
    uni.values.smokeTemp.set([value]);
    clearPressureRefreshSmoke = true;
  }, "Smoke source temperature (1=ambient)");
}

// Preset controls
{
  function autoUpdate(autoUpdate) {
    if (autoUpdate) refreshPreset(true);
  }
  let doAutoUpdate = true;
  gui.addGroup("presets", "Presets");

  gui.addNumericInput("radius", true, "Radius", { min: 0, max: 128, step: 1, val: sharedSettings.radius, float: 0 }, "presets", (value) => {sharedSettings.radius = value; autoUpdate(doAutoUpdate)});

  gui.addNumericInput("rotation", true, "AoA/Rot", { min: -90, max: 90, step: 1, val: 20, float: 0 }, "presets", (value) => {presetSettings.Prism.rot = presetSettings.FlatWing.AoA = toRad(value); autoUpdate(doAutoUpdate)});
  gui.addNumericInput("width", true, "Width", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "presets", (value) => {presetSettings.Prism.width = presetSettings.FlatWing.width = value; autoUpdate(doAutoUpdate)});
  gui.addRadioOptions("shape", ["circular", "square", "linear"], "circular", "presets", {}, (value) => {presetSettings.Aperture.shape = shapes[value]; autoUpdate(doAutoUpdate)});
  gui.addNumericInput("nSides", true, "n sides", { min: 3, max: 24, step: 1, val: 3, float: 0 }, "presets", (value) => {presetSettings.Prism.n = value; autoUpdate(doAutoUpdate)});

  gui.addNumericInput("chord", true, "Chord", { min: 8, max: 128, step: 1, val: 64, float: 0 }, "presets", (value) => {presetSettings.FlatWing.chord = value; autoUpdate(doAutoUpdate)});
  gui.addNumericInput("thickness", true, "Thickness", { min: 2, max: 10, step: 1, val: 3, float: 0 }, "presets", (value) => {presetSettings.FlatWing.thickness = value; autoUpdate(doAutoUpdate)});

  gui.addNumericInput("slitWidth", true, "Slit width", { min: 3, max: 512, step: 1, val: 8, float: 0 }, "presets", (value) => {presetSettings.DoubleSlit.slitWidth = value; autoUpdate(doAutoUpdate)});
  gui.addNumericInput("slitSpacing", true, "Slit spacing", { min: 0, max: 512, step: 1, val: 32, float: 0 }, "presets", (value) => {presetSettings.DoubleSlit.slitSpacing = value; autoUpdate(doAutoUpdate)});
  gui.addNumericInput("slitHeight", true, "Slit height", { min: 0, max: 512, step: 1, val: 64, float: 0 }, "presets", (value) => {presetSettings.DoubleSlit.slitHeight = value; autoUpdate(doAutoUpdate)});
  
  gui.addNumericInput("taperAngle", true, "Taper angle", { min: -60, max: 60, step: 1, val: -30, float: 0 }, "presets", (value) => {presetSettings.Aperture.taperAngle = toRad(-value); autoUpdate(doAutoUpdate)});
  gui.addCheckbox("invert", "Invert barrier", false, "presets", (checked) => presetSettings.Aperture.invert = checked);

  gui.addNumericInput("barrierThickness", true, "Thickness", { min: 1, max: 16, step: 1, val: 16, float: 0 }, "presets", (value) => {barrierThickness = value; autoUpdate(doAutoUpdate)});
  gui.addNumericInput("xOffset", true, "X Offset", { min: 0, max: 1, step: .01, val: 0.2, float: 2 }, "presets", (value) => {presetXOffset = Math.round(value * simulationDomain[0]); autoUpdate(doAutoUpdate)});

  gui.addDropdown("presetSelect", "Select preset", ["Prism", "FlatWing", "Aperture", "DoubleSlit"], "presets", {
    "Prism": ["radius", "rotation", "nSides", "width"],
    "FlatWing": ["chord", "rotation", "thickness", "width"],
    "Aperture": ["shape", "radius", "invert", "barrierThickness", "taperAngle"],
    "DoubleSlit": ["slitWidth", "slitSpacing", "slitHeight", "barrierThickness"],
  });

  gui.addCheckbox("doAutoUpdate", "Auto update", true, "presets", (checked) => doAutoUpdate = checked);

  gui.addButton("updatePreset", "Load preset", false, "presets", () => refreshPreset(false));
  gui.addButton("clearUpdatePreset", "Clear & load", false, "presets", () => refreshPreset(true));
  gui.addButton("clearPreset", "Clear", true, "presets", () => updateBarrierTexture(true));
}

// Visualization controls
{
  gui.addGroup("visCtrl", "Visualization controls");
  gui.addNumericInput("globalAlpha", true, "Global alpha", { min: 0, max: 5, step: 0.1, val: globalAlpha, float: 1 }, "visCtrl", (value) => {
    globalAlpha = value;
    uni.values.globalAlpha.set([value]);
  }, "Global alpha multiplier");
  gui.addNumericInput("rayDtMult", true, "Ray dt mult", { min: 0.1, max: 5, step: 0.1, val: 1.5, float: 1 }, "visCtrl", (value) => uni.values.rayDtMult.set([value]), "Raymarching step multipler; higher has better visual quality, lower has better performance");
  gui.addNumericInput("visMult", true, "Value multiplier", { min: 0.1, max: 5, step: 0.1, val: 1, float: 1 }, "visCtrl", (value) => uni.values.visMult.set([value]));
  gui.addCheckbox("showBarriers", "Show barriers", true, "visCtrl", (checked) => {
    if (checked) options |= 1;
    else options &= ~1;
    uni.values.options.set([options]);
  });
  gui.addCheckbox("renderIsosurface", "Render isosurface", false, "visCtrl", (checked) => {
    if (checked) {
      options |= (1 << 1);
      uni.values.globalAlpha.set([0]);
      gui.io.globalAlpha.value = 0;
    } else {
      options &= ~(1 << 1);
      uni.values.globalAlpha.set([globalAlpha]);
      gui.io.globalAlpha.value = globalAlpha;
    }
    uni.values.options.set([options]);
  });
  gui.addNumericInput("isoMin", true, "Isosurface min", { min: 0.1, max: 0.5, step: 0.1, val: 0.5, float: 1 }, "visCtrl", (value) => {
    uni.values.isoMin.set([value]);
    gui.io.isoMax.min = value + 0.1;
    if (gui.io.isoMax.value < value + 0.1) {
      uni.values.isoMax.set([value + 0.1]);
      gui.io.isoMax.value = value + 0.1;
    }
  });
  gui.addNumericInput("isoMax", true, "Isosurface max", { min: 0.6, max: 5, step: 0.1, val: 0.6, float: 1 }, "visCtrl", (value) => {
    uni.values.isoMax.set([value]);
    gui.io.isoMin.max = value - 0.1;
    if (gui.io.isoMin.value > value - 0.1) {
      uni.values.isoMin.set([value - 0.1]);
      gui.io.isoMin.value = value - 0.1;
    }
  });
}

{
  gui.addGroup("lightCtrl", "Lighting controls");
  gui.addCheckbox("enableLighting", "Enable lighting", false, "lightCtrl", (checked) => {
    if (checked) options |= (1 << 2);
    else options &= ~(1 << 2);
    uni.values.options.set([options]);
  });
  gui.addNumericInput("lightAzimuth", true, "Azimuth", { min: 0, max: 360, step: 1, val: 45, float: 0 }, "lightCtrl", (value) => {
    lightDir = vec3.normalize(sphericalToCartesian(toRad(value), Math.asin(lightDir[1]), 1));
    uni.values.lightDir.set(lightDir);
  });
  gui.addNumericInput("lightElevation", true, "Elevation", { min: -90, max: 90, step: 1, val: 45, float: 0 }, "lightCtrl", (value) => {
    lightDir = vec3.normalize(sphericalToCartesian(Math.atan2(lightDir[0], lightDir[2]), toRad(value), 1));
    uni.values.lightDir.set(lightDir);
  });
  gui.addNumericInput("lightIntensity", true, "Intensity", { min: 0, max: 10, step: 0.1, val: lightIntensity, float: 1 }, "lightCtrl", (value) => {
    lightIntensity = value;
    uni.values.lightColor.set(vec3.scale(lightColor, lightIntensity));
  });
  gui.addNumericInput("lightColorR", true, "Red", { min: 0, max: 1, step: 0.01, val: lightColor[0], float: 2 }, "lightCtrl", (value) => {
    lightColor[0] = value;
    uni.values.lightColor.set(vec3.scale(lightColor, lightIntensity));
  });
  gui.addNumericInput("lightColorG", true, "Green", { min: 0, max: 1, step: 0.01, val: lightColor[1], float: 2 }, "lightCtrl", (value) => {
    lightColor[1] = value;
    uni.values.lightColor.set(vec3.scale(lightColor, lightIntensity));
  });
  gui.addNumericInput("lightColorB", true, "Blue", { min: 0, max: 1, step: 0.01, val: lightColor[2], float: 2 }, "lightCtrl", (value) => {
    lightColor[2] = value;
    uni.values.lightColor.set(vec3.scale(lightColor, lightIntensity));
  });
  gui.addNumericInput("ambientIntensity", true, "Ambient", { min: 0, max: 10, step: 0.1, val: ambientIntensity, float: 1 }, "lightCtrl", (value) => {
    ambientIntensity = value;
    uni.values.ambientIntensity.set([ambientIntensity]);
  });
}

// Camera keybinds
gui.addGroup("camKeybinds", "Camera controls", `
  <div>
    Orbit: leftclick / arrows
    <br>
    Pan: rightclick / wasdgv
    <br>
    Zoom: scroll / fc
    <br>
    FOV zoom: ctrl+scroll / ctrl+fc
    <br>
    FOV: alt+scroll / alt+fc
    <br>
    Reset view: middleclick / space
    <br>
    Reset FOV: ctrl+middleclick / ctrl+space
  </div>
`);

// Extra info
gui.addGroup("guiControls", "GUI controls", `
  <div>
    Click on section titles to expand/collapse
    <br>
    Hover on input labels for more info if applicable
    <br>
    Click to toggle between raw number and slider type input
    <br>
  </div>
`);


// requestAnimationFrame id, fps update id
let rafId, perfIntId;


// timing
let jsTime = 0, lastFrameTime = performance.now(), deltaTime = 10, fps = 0,
  advectionComputeTime = 0, velDivComputeTime = 0, pressureComputeTime = 0, projectionComputeTime = 0, renderTime = 0;

// handle resizing
window.onresize = window.onload = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  camera.updateMatrix();
  uni.values.resolution.set([canvas.width, canvas.height]);
  gui.io.res([window.innerWidth, window.innerHeight]);
};

/**
 * Generates a random number within a range
 * @param {Number} min Lower bound, inclusive
 * @param {Number} max Upper bound, exclusive
 * @returns Random number between [min, max)
 */
const randRange = (min, max) => Math.random() * (max - min) + min;

/**
 * Generates a random number within a range of 0-max
 * @param {Number} max Upper bound, exclusive
 * @returns Random number between [0, max)
 */
const randMax = (max) => Math.random() * max;

/**
 * 
 * @param {Number} x x coordinate
 * @param {Number} y y coordinate
 * @param {Number} z z coordinate
 * @returns Linear index within simulation domain
 */
const index3d = (x, y, z) => x + simulationDomain[0] * (y + z * simulationDomain[1]);