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
  const f32filterable = adapter.features.has("float32-filterable");
  const shaderf16 = adapter.features.has("shader-f16");

  // compute workgroup size 16*8*8 | 32*8*4 | 64*4*4 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16*4*4 = 256 threads
  const largeWg = maxComputeInvocationsPerWorkgroup >= 1024;
  const [wg_x, wg_y, wg_z] = largeWg ? [16, 8, 8] : [16, 4, 4];

  if (!gpuInfo) {
    gui.addGroup("deviceInfo", "Device info", `
<pre><span ${!largeWg ? "class='warn'" : ""}>maxComputeInvocationsPerWorkgroup: ${maxComputeInvocationsPerWorkgroup}
workgroup: [${wg_x}, ${wg_y}, ${wg_z}]</span>
maxBufferSize: ${maxBufferSize}
f32filterable: ${f32filterable}
shader-f16: ${shaderf16}
</pre>
    `);
    gpuInfo = true;
  }

  device = await adapter?.requestDevice({
    requiredFeatures: [
      ...(adapter.features.has("timestamp-query") ? ["timestamp-query"] : []),
      ...(f32filterable ? ["float32-filterable"] : []),
      // ...(shaderf16 ? ["shader-f16"] : []),
    ],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: maxComputeInvocationsPerWorkgroup,
      maxBufferSize: maxBufferSize,
    }
  });
  device.addEventListener('uncapturederror', event => {
    if (event.error.message.includes("max buffer size limit"))
      alert(`Max buffer size exceeded. Your device supports max size ${maxBufferSize}, specified size ${simVoxelCount() * 4}`);
    // else alert(msg);
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

  const newTexture = (name, format = "r32float", storage = true) => device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
    label: `${name} texture`
  });

  // staggered grids?
  storage.velTex0 = newTexture("vel0", "rgba32float"); // try rgba16f using shader-f16 feature
  storage.velTex1 = newTexture("vel1", "rgba32float");
  storage.divTex = newTexture("divergence");
  storage.pressureTex = newTexture("pressure");
  // smoke + temp
  storage.smokeTemp0 = newTexture("smokeTemp0", "rg32float");
  storage.smokeTemp1 = newTexture("smokeTemp1", "rg32float");
  storage.curlTex = newTexture("curl");
  storage.barrierTex = newTexture("barrier", "r8unorm", false);

  // const velData = new Float32Array(simVoxelCount * 4).fill(0);
  // for (let z = 0; z < simulationDomain[2]; z++) {
  //   for (let y = 0; y < simulationDomain[1]; y++) {
  //     for (let x = 0; x < simulationDomain[0]; x++) {
  //       const index = x + y * simulationDomain[0] + z * simulationDomain[0] * simulationDomain[1];
  //       if (x <= 60 && x >= 0 && (y-simulationDomain[1]/2)**2 + (z-simulationDomain[2]/2)**2 < 16**2) {
  //         // initial velocity field
  //         velData[index * 4 + 0] = 2;
  //         velData[index * 4 + 1] = 0;
  //         velData[index * 4 + 2] = 0;
  //         velData[index * 4 + 3] = 0;
  //       }
  //       if (x >= 256-60 && x < 256-20 && (y-simulationDomain[1]/2)**2 + (z-simulationDomain[2]/2)**2 < 10**2) {
  //         // initial velocity field
  //         velData[index * 4 + 0] = -1;
  //         velData[index * 4 + 1] = 0;
  //         velData[index * 4 + 2] = 0;
  //         velData[index * 4 + 3] = 0;
  //       }
  //     }
  //   }
  // }
  // device.queue.writeTexture(
  //   { texture: storage.velTex0 },
  //   velData,
  //   { offset: 0, bytesPerRow: simulationDomain[0] * 4 * 4, rowsPerImage: simulationDomain[1] },
  //   { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  // );

  updateBarrierTexture();

  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name) =>
    device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: shaderCode,
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
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.velTex0.createView() },
      { binding: 2, resource: storage.velTex1.createView() },
      { binding: 3, resource: storage.smokeTemp0.createView() },
      { binding: 4, resource: storage.smokeTemp1.createView() },
      // { binding: 5, resource: storage.pressureTex.createView() },
    ],
    label: "init compute bind group"
  });

  const clearPressureComputePipeline = newComputePipeline(clearPressureShaderCode, "clear pressure");

  const clearPressureComputeBindGroup = device.createBindGroup({
    layout: clearPressureComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: storage.pressureTex.createView() },
    ],
    label: "clear pressure compute bind group"
  });

  const advectComputePipeline = newComputePipeline(advectionShaderCode, "advection");

  const linSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
    addressModeW: "clamp-to-edge",
  });

  const advectComputeBindGroup = (velTexOld, velTexNew, smokeTexOld, smokeTexNew) => device.createBindGroup({
    layout: advectComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velTexOld.createView() },
      { binding: 2, resource: velTexNew.createView() },
      { binding: 3, resource: storage.barrierTex.createView() },
      { binding: 4, resource: linSampler },
      { binding: 5, resource: smokeTexOld.createView() },
      { binding: 6, resource: smokeTexNew.createView() },
      { binding: 7, resource: storage.pressureTex.createView() },
    ],
    label: "advection compute bind group"
  });

  const advectComputeBindGroups = [
    advectComputeBindGroup(storage.velTex0, storage.velTex1, storage.smokeTemp0, storage.smokeTemp1),
    advectComputeBindGroup(storage.velTex0, storage.velTex1, storage.smokeTemp1, storage.smokeTemp0),
    // advectComputeBindGroup(storage.velTex1, storage.velTex0)
  ];

  const velDivComputePipeline = newComputePipeline(velDivShaderCode, "velocity divergence");

  const velDivComputeBindGroup = (velTex) => device.createBindGroup({
    layout: velDivComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velTex.createView() },
      { binding: 2, resource: storage.divTex.createView() },
      { binding: 3, resource: storage.barrierTex.createView() },
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
      { binding: 3, resource: storage.barrierTex.createView() },
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
      { binding: 4, resource: storage.barrierTex.createView() },
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
  // const pressureComputeTimingHelpers = new Array(8).fill(new TimingHelper(device)); // pressure helpers
  const projectionComputeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  const wgDispatchSize = [
    Math.ceil(simulationDomain[0] / wg_x),
    Math.ceil(simulationDomain[1] / wg_y),
    Math.ceil(simulationDomain[2] / wg_z)
  ]

  let pingPongIndex = 0;
  const pressureGlobalIter = 4;

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
      const initComputePass = encoder.beginComputePass();
      initComputePass.setPipeline(initComputePipeline);
      initComputePass.setBindGroup(0, initComputeBindGroup);
      initComputePass.dispatchWorkgroups(...wgDispatchSize);
      initComputePass.end();
    }

    if (clearPressure) {
      clearPressure = false;
      const clearPressureComputePass = encoder.beginComputePass();
      clearPressureComputePass.setPipeline(clearPressureComputePipeline);
      clearPressureComputePass.setBindGroup(0, clearPressureComputeBindGroup);
      clearPressureComputePass.dispatchWorkgroups(...wgDispatchSize);
      clearPressureComputePass.end();
    }

    const run = dt > 0;

    if (run) {
      const advectionComputePass = advectionComputeTimingHelper.beginComputePass(encoder);
      advectionComputePass.setPipeline(advectComputePipeline);
      advectionComputePass.setBindGroup(0, advectComputeBindGroups[pingPongIndex]);
      advectionComputePass.dispatchWorkgroups(...wgDispatchSize);
      advectionComputePass.end();

      const velDivComputePass = velDivComputeTimingHelper.beginComputePass(encoder);
      velDivComputePass.setPipeline(velDivComputePipeline);
      velDivComputePass.setBindGroup(0, velDivComputeBindGroups[0]);
      velDivComputePass.dispatchWorkgroups(...wgDispatchSize);
      velDivComputePass.end();

      for (let i = 0; i < pressureGlobalIter; i++) {
        // const pressureComputePass = pressureComputeTimingHelpers[i].beginComputePass(encoder);
        const pressureComputePass = encoder.beginComputePass();
        pressureComputePass.setPipeline(pressureComputePipeline);
        pressureComputePass.setBindGroup(0, pressureComputeBindGroup);
        pressureComputePass.dispatchWorkgroups(...wgDispatchSize);
        pressureComputePass.end();
      }

      const projectionComputePass = projectionComputeTimingHelper.beginComputePass(encoder);
      projectionComputePass.setPipeline(projectionComputePipeline);
      projectionComputePass.setBindGroup(0, projectionComputeBindGroups[0]);
      projectionComputePass.dispatchWorkgroups(...wgDispatchSize);
      projectionComputePass.end();

      // const velDivComputePass2 = encoder.beginComputePass();
      // velDivComputePass2.setPipeline(velDivComputePipeline);
      // velDivComputePass2.setBindGroup(0, velDivComputeBindGroups[pingPongIndex]);
      // velDivComputePass2.dispatchWorkgroups(...wgDispatchSize);
      // velDivComputePass2.end();

      pingPongIndex = 1 - pingPongIndex;
    }

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroups[renderTextureIdx + (pingPong ? pingPongIndex : 0)]); // 0 = velocity, 2 = divergence, 3 = pressure, 4-5 = smoke
    // renderPass.setBindGroup(0, renderBindGroups[pingPongIndex + 4]);
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    if (run) {
      advectionComputeTimingHelper.getResult().then(gpuTime => advectionComputeTime += (gpuTime / 1e6 - advectionComputeTime) / filterStrength);
      velDivComputeTimingHelper.getResult().then(gpuTime => velDivComputeTime += (gpuTime / 1e6 - velDivComputeTime) / filterStrength);
      // let pressureTime = 0;
      // pressureComputeTimingHelpers.forEach(e => e.getResult().then(gpuTime => pressureTime += gpuTime / 1e6));
      // pressureComputeTime = (pressureTime - pressureComputeTime) / filterStrength
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
    gui.io.computeTime(advectionComputeTime + velDivComputeTime + pressureComputeTime + projectionComputeTime);
    // gui.io.boundaryTime(boundaryComputeTime);
    gui.io.renderTime(renderTime);
  }, 100);

  camera.updatePosition();

  uni.values.dt.set([dt]);
  uni.values.volSize.set(simulationDomain);
  uni.values.volSizeNorm.set(simulationDomainNorm);
  uni.values.rayDtMult.set([2]);
  uni.values.resolution.set([canvas.width, canvas.height]);
  uni.values.vInflow.set([3]);
  uni.values.pressureLocalIter.set([4]); // 2-8 typical, 2-4 best according to chatgpt
  uni.values.SORomega.set([1]);
  // uni.values.vectorVis.set([0]);
  uni.values.globalAlpha.set([2]);
  uni.values.smokePos.set(smokePos);
  uni.values.smokeHalfSize.set(smokeHalfSize);

  rafId = requestAnimationFrame(render);
}

const camera = new Camera(defaults);

main().then(() => refreshPreset(false));