const uni = new Uniforms();
uni.addUniform("invMatrix", "mat4x4f");   // inverse proj*view matrix

uni.addUniform("cameraPos", "vec3f");     // camera position in world space
uni.addUniform("dt", "f32");              // simulation time step

uni.addUniform("volSize", "vec3f");       // volume size in voxels
uni.addUniform("rayDtMult", "f32");       // raymarch sampling factor

uni.addUniform("volSizeNorm", "vec3f");   // normalized volume size (volSize / max(volSize))
uni.addUniform("vInflow", "f32");         // -x boundary inflow velocity

uni.addUniform("resolution", "vec2f");    // canvas resolution: x-width, y-height
uni.addUniform("pressureLocalIter", "f32"); // number of rbgs subiterations using local memory 
uni.addUniform("vectorVis", "f32");       // Whether to render vector or scalar field

uni.addUniform("smokePos", "vec2f");       // Smoke source center position
uni.addUniform("smokeHalfSize", "vec2f");  // Smoke source size in Y and Z

uni.addUniform("SORomega", "f32");        // successive overrelaxation omega
uni.addUniform("globalAlpha", "f32");     // global alpha multiplier

uni.finalize();

const storage = {
  velTex0: null,
  velTex1: null,
  energyTex: null,
  barrierTex: null,
};

let initialize = true, clearPressure = true;

let renderTextureIdx = 4, pingPong = true;

let dt = 1;
let oldDt;

let dtPerFrame = 1;

const sharedSettings = {
  radius: 24,
  thickness: 16,
};

// simulation domain size [x, y, z], ex. [384, 256, 256], [512, 256, 384]
const simulationDomain = [256, 192, 192]; //[384, 256, 256];//[768, 384, 384];
let newDomainSize = vec3.clone(simulationDomain);
let simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];

let yMidpt = Math.floor(simulationDomain[1] / 2);
let zMidpt = Math.floor(simulationDomain[2] / 2);

const simulationDomainNorm = simulationDomain.map(v => v / Math.max(...simulationDomain));
let barrierData = new Uint8Array(simulationDomain[0] * simulationDomain[1] * simulationDomain[2]).fill(1);

let cleared = false;

const smokePos = [yMidpt, zMidpt];
const smokeHalfSize = [16, 8];

/**
 * Resizes the simulation domain
 * @param {Array<Number>} newSize New simulation domain size
 */
function resizeDomain(newSize) {
  vec3.clone(newSize, simulationDomain);
  vec3.clone(simulationDomain.map(v => v / Math.max(...simulationDomain)), simulationDomainNorm);
  simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];
  barrierData = new Uint8Array(simVoxelCount).fill(1);
  yMidpt = Math.floor(simulationDomain[1] / 2);
  zMidpt = Math.floor(simulationDomain[2] / 2);
  camera.target = defaults.target = vec3.scale(simulationDomainNorm, 0.5);
}

/**
 * Refreshes the active preset
 */
function refreshPreset(clear = false) {
  if (clear) barrierData.fill(1);
  clearPressure = true;
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
  clearPressure = true;
}

function hardReset() {
  initialize = true;
  clearPressure = true;
  cancelAnimationFrame(rafId);
  clearInterval(perfIntId);
  if (!vec3.equals(simulationDomain, newDomainSize)) resizeDomain(newDomainSize);
  storage.barrierTex.destroy();
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
  gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfR");
  gui.addNumericOutput("boundaryTime", "Boundary", "ms", 2, "perfR");
  gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfR");
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
  gui.addDropdown("visType", "Visualization", ["Smoke", "Velocity", "Velocity magnitude", "Pressure", "Divergence"], "simCtrl", null, (value) => {
    uni.values.vectorVis.set([0]);
    pingPong = false;
    switch (value) {
      case "Smoke":
        renderTextureIdx = 4;
        pingPong = true;
        break;
      case "Velocity":
        uni.values.vectorVis.set([1]);
        renderTextureIdx = 0;
        break;
      case "Velocity magnitude":
        uni.values.vectorVis.set([2]);
        renderTextureIdx = 0;
        break;
      case "Pressure":
        renderTextureIdx = 3;
        break;
      case "Divergence":
        renderTextureIdx = 2;
        break;
    }
  });
  gui.addNumericInput("dt", true, "dt", { min: 0, max: 2, step: 0.01, val: dt, float: 2 }, "simCtrl", (newDt) => {
    if (oldDt) oldDt = newDt;
    else dt = newDt;
    uni.values.dt.set([dt]);
  }, "Simulation delta-time");
  gui.addNumericInput("inflowV", true, "Flow velocity", { min: 0, max: 5, step: 0.01, val: 3, float: 2 }, "simCtrl", (value) => uni.values.vInflow.set([value]));
  gui.addNumericInput("xSize", false, "X size (reinit)", { min: 8, max: 1024, step: 8, val: simulationDomain[0], float: 0 }, "simCtrl", (value) => newDomainSize[0] = value, "Requires reinitialization to apply");
  gui.addNumericInput("ySize", false, "Y size (reinit)", { min: 8, max: 512, step: 8, val: simulationDomain[1], float: 0 }, "simCtrl", (value) => newDomainSize[1] = value, "Requires reinitialization to apply");
  gui.addNumericInput("zSize", false, "Z size (reinit)", { min: 8, max: 512, step: 8, val: simulationDomain[2], float: 0 }, "simCtrl", (value) => newDomainSize[2] = value, "Requires reinitialization to apply");
  gui.addNumericInput("smokePY", true, "Y pos", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "simCtrl", (value) => {
    smokePos[0] = Math.round(value * simulationDomain[1]);
    uni.values.smokePos.set(smokePos);
  }, "Smoke source Y-coordinate normalized to simulation YZ plane, requires reinit");
  gui.addNumericInput("smokePZ", true, "Z pos", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "simCtrl", (value) => {
    smokePos[1] = Math.round(value * simulationDomain[2]);
    uni.values.smokePos.set(smokePos);
  }, "Smoke source Z-coordinate normalized to simulation YZ plane, requires reinit");
  gui.addNumericInput("smokeSY", true, "Y size", { min: 0, max: 1, step: 0.01, val: smokeHalfSize[0] * 2 / simulationDomain[1], float: 2 }, "simCtrl", (value) => {
    smokeHalfSize[0] = Math.ceil(value * simulationDomain[1] / 2);
    uni.values.smokeHalfSize.set(smokeHalfSize);
  }, "Smoke source Y size normalized to simulation YZ plane, requires reinit");
  gui.addNumericInput("smokeSZ", true, "Z size", { min: 0, max: 1, step: 0.01, val: smokeHalfSize[1] * 2 / simulationDomain[2], float: 2 }, "simCtrl", (value) => {
    smokeHalfSize[1] = Math.ceil(value * simulationDomain[2] / 2);
    uni.values.smokeHalfSize.set(smokeHalfSize);
  }, "Smoke source Z size normalized to simulation YZ plane, requires reinit");
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

// Preset controls
{
  gui.addGroup("presets", "Presets");

  gui.addNumericInput("radius", true, "Radius", { min: 0, max: 256, step: 1, val: sharedSettings.radius, float: 0 }, "presets", (value) => sharedSettings.radius = value);

  gui.addNumericInput("rotation", true, "AoA/Rot", { min: -90, max: 90, step: 1, val: 10, float: 0 }, "presets", (value) => presetSettings.Prism.rot = presetSettings.FlatWing.AoA = value.toRad());
  gui.addNumericInput("width", true, "Width", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "presets", (value) => presetSettings.Prism.width = presetSettings.FlatWing.width = value);
  gui.addRadioOptions("shape", ["circular", "square", "linear"], "circular", "presets", {}, (value) => presetSettings.Aperture.shape = shapes[value]);
  gui.addNumericInput("nSides", true, "n sides", { min: 3, max: 24, step: 1, val: 3, float: 0 }, "presets", (value) => presetSettings.Prism.n = value);

  gui.addNumericInput("chord", true, "Chord", { min: 8, max: 64, step: 1, val: 64, float: 0 }, "presets", (value) => presetSettings.FlatWing.chord = value);
  gui.addNumericInput("thickness", true, "Thickness", { min: 2, max: 10, step: 1, val: 3, float: 0 }, "presets", (value) => presetSettings.FlatWing.thickness = value);

  gui.addNumericInput("slitWidth", true, "Slit width", { min: 3, max: 512, step: 1, val: 8, float: 0 }, "presets", (value) => presetSettings.DoubleSlit.slitWidth = value);
  gui.addNumericInput("slitSpacing", true, "Slit spacing", { min: 0, max: 512, step: 1, val: 32, float: 0 }, "presets", (value) => presetSettings.DoubleSlit.slitSpacing = value);
  gui.addNumericInput("slitHeight", true, "Slit height", { min: 0, max: 512, step: 1, val: 64, float: 0 }, "presets", (value) => presetSettings.DoubleSlit.slitHeight = value);

  gui.addCheckbox("invert", "Invert barrier", false, "presets", (checked) => presetSettings.Aperture.invert = checked);

  gui.addNumericInput("barrierThickness", true, "Thickness", { min: 1, max: 16, step: 1, val: 2, float: 0 }, "presets", (value) => barrierThickness = value);
  gui.addNumericInput("xOffset", true, "X Offset", { min: 0, max: 512, step: 1, val: 16, float: 0 }, "presets", (value) => presetXOffset = value);

  gui.addDropdown("presetSelect", "Select preset", ["Prism", "FlatWing", "Aperture", "DoubleSlit"], "presets", {
    "Prism": ["radius", "rotation", "nSides", "width"],
    "FlatWing": ["chord", "rotation", "thickness", "width"],
    "Aperture": ["shape", "radius", "invert", "barrierThickness"],
    "DoubleSlit": ["slitWidth", "slitSpacing", "slitHeight", "barrierThickness"],
  });

  gui.addButton("updatePreset", "Load preset", false, "presets", () => refreshPreset(false));
  gui.addButton("clearUpdatePreset", "Clear & load", false, "presets", () => refreshPreset(true));
  gui.addButton("clearPreset", "Clear", true, "presets", () => updateBarrierTexture(true));
}

// Visualization controls
{
  gui.addGroup("visCtrl", "Visualization controls");
  gui.addNumericInput("globalAlpha", true, "Global alpha", { min: 0.1, max: 5, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => uni.values.globalAlpha.set([value]), "Global alpha multiplier");
  gui.addNumericInput("rayDtMult", true, "Ray dt mult", { min: 0.1, max: 5, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => uni.values.rayDtMult.set([value]), "Raymarching step multipler; higher has better visual quality, lower has better performance");
  gui.addCheckbox("energy", "Visualize energy", true, "visCtrl", (checked) => {
    energyFilterStrength = checked ? defaultEnergyFilterStrength : 0;
    uni.values.energyFilter.set([energyFilterStrength]);
  });
  gui.addNumericInput("plusXAlpha", true, "+X energy a", { min: 1, max: 5, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => uni.values.plusXAlpha.set([value]), "+X energy projection alpha multiplier");
  gui.addNumericInput("energyMult", true, "Energy mult", { min: 0.01, max: 5, step: 0.01, val: 1, float: 2 }, "visCtrl", (value) => uni.values.energyMult.set([value]), "Raw energy value multiplier before transfer function");
  gui.addNumericInput("energyFilter", true, "Energy filter", { min: 0, max: 3, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => {
    value = Math.pow(10, value);
    defaultEnergyFilterStrength = value;
    energyFilterStrength = gui.io.energy.checked ? defaultEnergyFilterStrength : 0;
    uni.values.energyFilter.set([value]);
  }, "Energy low pass filter strength");
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
 * Clamps a number between between specified values
 * @param {Number} min Lower bound to clamp
 * @param {Number} max Upper bound to clamp
 * @returns Original number clamped between min and max
 */
Number.prototype.clamp = function (min, max) { return Math.max(min, Math.min(max, this)) };

/**
 * Converts degrees to radians
 * @returns Degree value in radians
 */
Number.prototype.toRad = function () { return this * Math.PI / 180; }

/**
 * Converts radians to degrees
 * @returns Radian value in degrees
 */
Number.prototype.toDeg = function () { return this / Math.PI * 180; }

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