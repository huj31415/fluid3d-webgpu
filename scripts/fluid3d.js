let adapter, device;
let gpuInfo = false;

async function main() {

  if (device) device.destroy();

  // let maxComputeInvocationsPerWorkgroup, maxBufferSize, f32filterable;

  // WebGPU Setup
  // if (!device) {
  adapter = await navigator.gpu?.requestAdapter();

  const maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  const maxBufferSize = adapter.limits.maxBufferSize;
  const maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  const f32filterable = adapter.features.has("float32-filterable");
  const shaderf16 = adapter.features.has("shader-f16");
  
  if (!shaderf16 && !gpuInfo) {
    alert("shader-f16 feature not supported, using 32 bit float textures and reducing domain size");
    newDomainSize = [256,128,192];
    hardReset();
    // todo: adjust everything else to new size
  }

  const floatPrecision = shaderf16 ? 16 : 32;
  const f16header = shaderf16 ? `
enable f16;
// alias vec4h = vec4<f${floatPrecision}>;
// alias vec3h = vec3<f${floatPrecision}>;
// alias vec2h = vec2<f${floatPrecision}>;
` : "";

  const textureTier1 = adapter.features.has("texture-formats-tier1");
  if (!textureTier1 && !gpuInfo) alert("texture-formats-tier1 feature required");
  const textureTier2 = adapter.features.has("texture-formats-tier2");
  if (!textureTier2 && !gpuInfo) alert("texture-formats-tier2 unsupported, may reduce performance");

  // compute workgroup size 16*8*8 | 32*8*4 | 64*4*4 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16*4*4 = 256 threads
  const largeWg = maxComputeInvocationsPerWorkgroup >= 1024;
  const [wg_x, wg_y, wg_z] = largeWg ? [16, 8, 8] : [16, 4, 4];

  if (!gpuInfo) {
    gui.addGroup("deviceInfo", "Device info", `
<pre><span ${!largeWg ? "class='warn'" : ""}>maxComputeInvocationsPerWorkgroup: ${maxComputeInvocationsPerWorkgroup}
workgroup: [${wg_x}, ${wg_y}, ${wg_z}]</span>
maxBufferSize: ${maxBufferSize}
maxStorageBufferBindingSize: ${maxStorageBufferBindingSize}
f32filterable: ${f32filterable}
shader-f16: ${shaderf16}
texture-formats-tier1: ${textureTier1}
<span ${!textureTier2 ? "class='warn'" : ""}>texture-formats-tier2: ${textureTier2}</span>
</pre>
    `);
    gpuInfo = true;

    // return to implement the f32-only domain size change
    // because the dawn upload buffer is somehow 3.6GB even though total texture size is significantly smaller
    // last known to have worked in 2025-11, buffer size errors in 2026-02
    if (!shaderf16) return;
  }


  device = await adapter?.requestDevice({
    requiredFeatures: [
      ...(adapter.features.has("timestamp-query") ? ["timestamp-query"] : []),
      ...(f32filterable ? ["float32-filterable"] : []),
      ...(textureTier1 ? ["texture-formats-tier1"] : []),
      ...(textureTier2 ? ["texture-formats-tier2"] : []),
      ...(shaderf16 ? ["shader-f16"] : []),
    ],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: maxComputeInvocationsPerWorkgroup,
      maxBufferSize: maxBufferSize,
      maxStorageBufferBindingSize: maxStorageBufferBindingSize,
    }
  });
  device.addEventListener('uncapturederror', event => {
    const msg = event.error.message;
    if (msg.includes("max buffer size limit"))
      alert(`Max buffer size exceeded. Reduce the simulation domain size to decrease buffer size`);
    else {
      alert(msg);
    }
    cancelAnimationFrame(rafId);
    return;
  });

  // restart if device crashes
  device.lost.then((info) => {
    if (info.reason != "destroyed") {
      hardReset();
      console.warn("WebGPU device lost, reinitializing.");
    }
  });

  // }
  if (!device) {
    alert("Browser does not support WebGPU");
    document.body.textContent = "WebGPU is not supported in this browser.";
    return;
  }
  const context = canvas.getContext("webgpu");
  const swapChainFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: swapChainFormat,
  });

  // advect (v) -> jacobi diffusion (if viscous) (div,p) -> force
  // -> jacobi pressure (>20iter) -> pressure projection (save in local memory? probably not due to halo access for laplacian for projection step)

  const newTexture = (name, format = `r${floatPrecision}float`, copyDst = false, storage = true) => device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: format,
    usage: GPUTextureUsage.TEXTURE_BINDING | (copyDst ? GPUTextureUsage.COPY_DST : 0) | (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
    label: `${name} texture`
  });

  // staggered grids?
  storage.velTex0 = newTexture("vel0", `rgba${floatPrecision}float`);
  storage.velTexM = newTexture("velM", `rgba${floatPrecision}float`);
  storage.velTex1 = newTexture("vel1", `rgba${floatPrecision}float`);
  storage.divTex = newTexture("divergence");
  storage.pressureTex = newTexture("pressure", `r${textureTier2 ? floatPrecision : 32}float`);
  // smoke + temp
  storage.smokeTemp0 = newTexture("smokeTemp0", `rg${floatPrecision}float`);
  storage.smokeTempM = newTexture("smokeTempM", `rg${floatPrecision}float`);
  storage.smokeTemp1 = newTexture("smokeTemp1", `rg${floatPrecision}float`);
  storage.curlTex = newTexture("curl", `rgba${floatPrecision}float`);
  storage.barrierTex = newTexture("barrier", "r8unorm", true, false);
  // store bitmask for barriers in 6 directions
  storage.barrierMask = newTexture("barrierMask", "r8uint", false, true);

  updateBarrierTexture();

  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name) =>
    device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: f16header + shaderCode(floatPrecision, textureTier2 ? floatPrecision : 32),
          label: `${name} compute module`
        }),
        constants: {
          WG_X: wg_x,
          WG_Y: wg_y,
          WG_Z: wg_z
        },
        entryPoint: 'main'
      },
      label: `${name} compute pipeline`
    });

  const initComputePipeline = newComputePipeline(initShaderCode, "init");

  const initComputeBindGroup = device.createBindGroup({
    layout: initComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.velTex0.createView() },
      { binding: 2, resource: storage.velTex1.createView() },
      { binding: 3, resource: storage.smokeTemp0.createView() },
      { binding: 4, resource: storage.smokeTemp1.createView() },
      // { binding: 5, resource: storage.pressureTex.createView() },
    ],
    label: "init compute bind group"
  });

  const clearPressureRefreshSmokeComputePipeline = newComputePipeline(clearPressureRefreshSmokeShaderCode, "clear pressure");

  const clearPressureRefreshSmokeComputeBindGroup = device.createBindGroup({
    layout: clearPressureRefreshSmokeComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.pressureTex.createView() },
      { binding: 2, resource: storage.smokeTemp0.createView() },
      { binding: 3, resource: storage.smokeTemp1.createView() },
    ],
    label: "clear pressure compute bind group"
  });

  const barrierMaskComputePipeline = newComputePipeline(barrierMaskShaderCode, "barrier mask");

  const barrierMaskComputeBindGroup = device.createBindGroup({
    layout: barrierMaskComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.barrierTex.createView() },
      { binding: 2, resource: storage.barrierMask.createView() },
    ],
    label: "barrier mask compute bind group"
  });

  const linSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
    addressModeW: "clamp-to-edge",
  });

  const semiLagrangianAdvectComputePipeline = newComputePipeline(semiLagrangianAdvectionShaderCode, "advection");

  const semiLagrangianAdvectComputeBindGroup = (velTexOld, velTexNew, smokeTexOld, smokeTexNew) => device.createBindGroup({
    layout: semiLagrangianAdvectComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velTexOld.createView() },
      { binding: 2, resource: velTexNew.createView() },
      { binding: 3, resource: storage.barrierTex.createView() },
      { binding: 4, resource: linSampler },
      { binding: 5, resource: smokeTexOld.createView() },
      { binding: 6, resource: smokeTexNew.createView() },
      // { binding: 7, resource: storage.pressureTex.createView() },
    ],
    label: "advection compute bind group"
  });

  const semiLagrangianAdvectComputeBindGroups = [
    semiLagrangianAdvectComputeBindGroup(storage.velTex0, storage.velTex1, storage.smokeTemp0, storage.smokeTemp1),
    semiLagrangianAdvectComputeBindGroup(storage.velTex0, storage.velTex1, storage.smokeTemp1, storage.smokeTemp0),
    // maccormack advection stage 1
    semiLagrangianAdvectComputeBindGroup(storage.velTex0, storage.velTexM, storage.smokeTemp0, storage.smokeTempM),
    semiLagrangianAdvectComputeBindGroup(storage.velTex0, storage.velTexM, storage.smokeTemp1, storage.smokeTempM)
  ];

  const mcAdvectComputePipeline = newComputePipeline(mcAdvectionShaderCode2, "mcAdvection2");

  const mcAdvectComputeBindGroup = (velTexOld, velTexNew, smokeTexOld, smokeTexNew) => device.createBindGroup({
    layout: mcAdvectComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velTexOld.createView() },
      { binding: 2, resource: storage.velTexM.createView() },
      { binding: 3, resource: velTexNew.createView() },
      { binding: 4, resource: storage.barrierTex.createView() },
      { binding: 5, resource: linSampler },
      { binding: 6, resource: smokeTexOld.createView() },
      { binding: 7, resource: storage.smokeTempM.createView() },
      { binding: 8, resource: smokeTexNew.createView() },
    ],
    label: "mcAdvect2 compute bind group"
  });

  const mcAdvectComputeBindGroups = [
    mcAdvectComputeBindGroup(storage.velTex0, storage.velTex1, storage.smokeTemp0, storage.smokeTemp1),
    mcAdvectComputeBindGroup(storage.velTex0, storage.velTex1, storage.smokeTemp1, storage.smokeTemp0),
  ]

  const velDivComputePipeline = newComputePipeline(velDivShaderCode, "velocity divergence");

  const velDivComputeBindGroup = (velTex) => device.createBindGroup({
    layout: velDivComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velTex.createView() },
      { binding: 2, resource: storage.divTex.createView() },
      { binding: 3, resource: storage.curlTex.createView() },
      { binding: 4, resource: storage.barrierMask.createView() },
      // { binding: 4, resource: storage.barrierTex.createView() },
    ],
    label: "velocity divergence compute bind group"
  });

  const velDivComputeBindGroups = [
    velDivComputeBindGroup(storage.velTex1),
    // velDivComputeBindGroup(storage.velTex0)
  ];

  const pressureComputePipeline = newComputePipeline(pressureShaderCode, "pressure");

  const pressureComputeBindGroup = device.createBindGroup({
    layout: pressureComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.divTex.createView() },
      { binding: 2, resource: storage.pressureTex.createView() },
      { binding: 3, resource: storage.barrierMask.createView() },
      // { binding: 3, resource: storage.barrierTex.createView() },
    ],
    label: "pressure compute bind group"
  });

  const projectionComputePipeline = newComputePipeline(projectionShaderCode, "pressure projection");

  const projectionComputeBindGroup = (velTexOld, velTexNew) => device.createBindGroup({
    layout: projectionComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velTexOld.createView() },
      { binding: 2, resource: velTexNew.createView() },
      { binding: 3, resource: storage.pressureTex.createView() },
      // { binding: 4, resource: storage.barrierTex.createView() },
      { binding: 4, resource: storage.barrierMask.createView() },
    ],
    label: "pressure projection compute bind group"
  });

  const projectionComputeBindGroups = [
    projectionComputeBindGroup(storage.velTex1, storage.velTex0),
    // projectionComputeBindGroup(storage.velTex0, storage.velTex1)
  ];

  const renderModule = device.createShaderModule({
    code: renderShaderCode,
    label: "render module"
  });

  const filter = f32filterable ? "linear" : "nearest";
  const f32sampler = device.createSampler({
    magFilter: filter,
    minFilter: filter,
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
    addressModeW: "clamp-to-edge",
  });

  const renderPipeline = device.createRenderPipeline({
    label: '3d volume rendering pipeline',
    layout: 'auto',
    vertex: { module: renderModule },
    fragment: {
      module: renderModule,
      targets: [{ format: swapChainFormat }],
      constants: {}
    }
  });

  const renderBindGroup = (tex) => device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex.createView() },
      { binding: 2, resource: storage.barrierTex.createView() },
      { binding: 3, resource: linSampler },
      { binding: 4, resource: f32sampler },
    ],
  });

  const renderBindGroups = [
    renderBindGroup(storage.velTex1),
    renderBindGroup(storage.velTex0),
    renderBindGroup(storage.divTex),
    renderBindGroup(storage.pressureTex),
    renderBindGroup(storage.smokeTemp0),
    renderBindGroup(storage.smokeTemp1),
    renderBindGroup(storage.curlTex),
  ];

  const renderPassDescriptor = {
    label: 'render pass',
    colorAttachments: [
      {
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ]
  };
  const filterStrength = 50;

  const advectionComputeTimingHelper = new TimingHelper(device);
  const velDivComputeTimingHelper = new TimingHelper(device);
  // const pressureComputeTimingHelper = new TimingHelper(device);
  const pressureComputeTimingHelpers = new Array(pressureGlobalIter);//.fill(new TimingHelper(device)); // pressure helpers
  for (let i = 0; i < pressureGlobalIter; i++) {
    pressureComputeTimingHelpers[i] = new TimingHelper(device);
  }
  const projectionComputeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  const wgDispatchSize = [
    Math.ceil(simulationDomain[0] / wg_x),
    Math.ceil(simulationDomain[1] / wg_y),
    Math.ceil(simulationDomain[2] / wg_z)
  ]

  let pingPongIndex = 0;
  let pressureTime = 0;

  function createComputePass(pass, pipeline, bindGroup) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...wgDispatchSize);
    pass.end();
  }

  function render() {
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    const speedMultiplier = Math.min(deltaTime, 50);
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    camera.handleInputs(speedMultiplier);

    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    // device.queue.writeBuffer(uniformBuffer, 0, uni.uniformData);
    uni.update(device.queue);

    const encoder = device.createCommandEncoder();

    if (initialize) {
      initialize = false;
      createComputePass(encoder.beginComputePass(), initComputePipeline, initComputeBindGroup);
    }

    if (clearPressureRefreshSmoke) {
      clearPressureRefreshSmoke = false;
      createComputePass(encoder.beginComputePass(), clearPressureRefreshSmokeComputePipeline, clearPressureRefreshSmokeComputeBindGroup);
    }

    if (updateBarrierMask) {
      updateBarrierMask = false;
      createComputePass(encoder.beginComputePass(), barrierMaskComputePipeline, barrierMaskComputeBindGroup);
    }

    const run = dt > 0;

    if (run) {
      createComputePass(velDivComputeTimingHelper.beginComputePass(encoder), velDivComputePipeline, velDivComputeBindGroups[0]);

      for (let i = 0; i < pressureGlobalIter; i++) {
        createComputePass(pressureComputeTimingHelpers[i].beginComputePass(encoder), pressureComputePipeline, pressureComputeBindGroup);
        // need to run a global iteration after
      }

      createComputePass(projectionComputeTimingHelper.beginComputePass(encoder), projectionComputePipeline, projectionComputeBindGroups[0]);

      switch (advectionMode) {
        case 0:
          createComputePass(encoder.beginComputePass(), semiLagrangianAdvectComputePipeline, semiLagrangianAdvectComputeBindGroups[pingPongIndex + 2]);
          createComputePass(advectionComputeTimingHelper.beginComputePass(encoder), mcAdvectComputePipeline, mcAdvectComputeBindGroups[pingPongIndex]);
          break;
        case 1:
          createComputePass(advectionComputeTimingHelper.beginComputePass(encoder), semiLagrangianAdvectComputePipeline, semiLagrangianAdvectComputeBindGroups[pingPongIndex]);
          break;
      }
      pingPongIndex = 1 - pingPongIndex;
    }

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroups[renderTextureIdx + (pingPong ? pingPongIndex : 0)]); // 0 = velocity, 2 = divergence, 3 = pressure, 4-5 = smoke, 6 = curl
    // renderPass.setBindGroup(0, renderBindGroups[pingPongIndex + 4]);
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    if (run) {
      advectionComputeTimingHelper.getResult().then(gpuTime => advectionComputeTime += (gpuTime / 1e6 - advectionComputeTime) / filterStrength);
      velDivComputeTimingHelper.getResult().then(gpuTime => velDivComputeTime += (gpuTime / 1e6 - velDivComputeTime) / filterStrength);

      pressureComputeTimingHelpers.forEach(e => e.getResult().then(gpuTime => pressureTime += gpuTime));
      pressureComputeTime += (pressureTime / 1e6 - pressureComputeTime) / filterStrength;
      pressureTime = 0;

      projectionComputeTimingHelper.getResult().then(gpuTime => projectionComputeTime += (gpuTime / 1e6 - projectionComputeTime) / filterStrength);
    } else {
      advectionComputeTime = velDivComputeTime = pressureComputeTime = projectionComputeTime = 0;
    }
    renderTimingHelper.getResult().then(gpuTime => renderTime += (gpuTime / 1e6 - renderTime) / filterStrength);

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
  }

  perfIntId = setInterval(() => {
    gui.io.fps(fps);
    gui.io.jsTime(jsTime);
    gui.io.frameTime(deltaTime);
    gui.io.computeTime(advectionComputeTime + velDivComputeTime + projectionComputeTime + pressureComputeTime);
    gui.io.renderTime(renderTime);
    gui.io.vDivTime(velDivComputeTime);
    gui.io.pressureTime(pressureComputeTime);
    gui.io.vProjTime(projectionComputeTime);
    gui.io.advectionTime(advectionComputeTime);
  }, 100);

  camera.updatePosition();

  uni.values.dt.set([dt]);
  uni.values.volSize.set(simulationDomain);
  uni.values.volSizeNorm.set(simulationDomainNorm);
  uni.values.resolution.set([canvas.width, canvas.height]);
  uni.values.SORomega.set([1.6]);
  // uni.values.visMode.set([0]);
  uni.values.smokePos.set(smokePos);
  uni.values.smokeHalfSize.set(smokeHalfSize);

  rafId = requestAnimationFrame(render);
}

const camera = new Camera(defaults);

camera.target = defaults.target = vec3.scale(simulationDomainNorm, 0.5);

uni.values.vInflow.set([2]);
uni.values.smokeTemp.set([1]);
uni.values.options.set([options]);
uni.values.pressureLocalIter.set([1]); // 2-8 typical, 2-4 best according to chatgpt
uni.values.globalAlpha.set([globalAlpha]);
uni.values.rayDtMult.set([1.5]);
uni.values.visMult.set([1]);
uni.values.isoMin.set([0.5]);
uni.values.isoMax.set([0.6]);
uni.values.lightDir.set(lightDir);
uni.values.ambientIntensity.set([ambientIntensity]);
uni.values.lightColor.set(vec3.scale(lightColor, lightIntensity));
uni.values.phaseG.set([0.5]);
uni.values.absorption.set([10]);
// uni.values.scattering.set([5]);

main().then(() => refreshPreset(false));