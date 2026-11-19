import { cameras, lines, passes, targets, type CameraBase } from '@typegpu/kit';
import { caps, joins } from '@typegpu/kit/geometry';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';

const SIZE = 64;

/** One of an 8-bit texture's channels from 0 to 1, row by row. */
async function readChannel(root: TgpuRoot, texture: GPUTexture, channel: number) {
  const readback = root.device.createBuffer({
    size: SIZE * 4 * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: SIZE * 4 }, [SIZE, SIZE]);
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange().slice(0));
  return Float32Array.from({ length: SIZE * SIZE }, (_, i) => bytes[i * 4 + channel] / 255);
}

const readRed = (root: TgpuRoot, texture: GPUTexture) => readChannel(root, texture, 0);

/** A `depth32float` texture's depth, row by row. */
async function readDepth(root: TgpuRoot, texture: GPUTexture) {
  const readback = root.device.createBuffer({
    size: SIZE * 4 * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer(
    { texture, aspect: 'depth-only' },
    { buffer: readback, bytesPerRow: SIZE * 4 },
    [SIZE, SIZE],
  );
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  return new Float32Array(readback.getMappedRange().slice(0));
}

/**
 * Draws `set` alone through `view` into a cleared 64 by 64 target of `pixelRatio` and reads its
 * red channel.
 */
async function drawn(
  root: TgpuRoot,
  set: lines.Lines,
  view: CameraBase = flat(root),
  pixelRatio = 1,
) {
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE, pixelRatio },
    { color: { format: 'rgba8unorm' } },
  );
  const into = passes.create(root, screen, { color: passes.color }).pipe(view.inject());
  const encoder = root.createCommandEncoder();
  into.record(encoder, into.add(set.drawing));
  encoder.submit();
  const red = await readRed(root, root.unwrap(screen.textures.color));
  into.destroy();
  screen.destroy();
  return red;
}

const column = (red: Float32Array, x: number) =>
  Array.from({ length: SIZE }, (_, y) => red[y * SIZE + x]);
const sum = (values: ArrayLike<number>) =>
  Array.prototype.reduce.call(values, (a, b) => a + b, 0) as number;

/** A flat camera ten units tall over the 64 pixel target. */
const flat = (root: TgpuRoot) => {
  const view = cameras.flat(root, { size: { width: SIZE, height: SIZE }, height: 10 });
  view.jump();
  return view;
};

it('draws a line its width in points, times the pass pixel ratio, with partly covered edges', async () => {
  const root = await gpuRoot();
  const paths = lines.create(root, { capacity: 1, length: 2, units: 'points' });
  paths.write(0, [-4, 0.03, 0, 4, 0.03, 0], { width: 3 });

  for (const ratio of [1, 2]) {
    const across = column(await drawn(root, paths, flat(root), ratio), 32);
    // A band 3 points wide covers 3 pixels per point of each column
    expect(sum(across)).toBeCloseTo(3 * ratio, 0);
    expect(across.filter((value) => value > 0.02 && value < 0.98).length).toBeGreaterThan(0);
  }
  root.destroy();
});

it('covers each pixel by the share of it the line covers', async () => {
  const root = await gpuRoot();
  const paths = lines.create(root, { capacity: 1, length: 2, units: 'points' });
  paths.write(0, [-4, 0, 0, 4, 0, 0], { width: 3 });

  // Three points about the boundary between rows 31 and 32
  const across = column(await drawn(root, paths), 32).slice(29, 35);
  for (const [got, want] of across.map((value, i) => [value, [0, 0.5, 1, 1, 0.5, 0][i]])) {
    expect(got).toBeCloseTo(want!, 1);
  }
  root.destroy();
});

it('draws a line narrower than minWidth at that width, fading it by how much narrower it is', async () => {
  const root = await gpuRoot();
  const hairline = lines.create(root, { capacity: 1, length: 2, units: 'points' });
  hairline.write(0, [-4, 0.03, 0, 4, 0.03, 0], { width: 0.25 });

  const across = column(await drawn(root, hairline), 32);
  expect(sum(across)).toBeCloseTo(0.25, 1);
  expect(Math.max(...across)).toBeLessThan(0.3);
  root.destroy();
});

it('takes minWidth in pixels, so a 2x screen draws the hairline one pixel wide', async () => {
  const root = await gpuRoot();
  // Half a pixel above the middle, so a one-pixel line covers one row
  const y = 0.5 / 6.4;
  const inPoints = lines.create(root, { capacity: 1, length: 2, units: 'points' });
  const inPixels = lines.create(root, {
    capacity: 1,
    length: 2,
    units: 'points',
    minWidth: { pixels: 1 },
  });
  for (const set of [inPoints, inPixels]) set.write(0, [-4, y, 0, 4, y, 0], { width: 0.25 });

  const points = column(await drawn(root, inPoints, flat(root), 2), 32);
  const pixels = column(await drawn(root, inPixels, flat(root), 2), 32);
  // A quarter point is half a pixel either way, spread over two pixels or one
  expect(sum(points)).toBeCloseTo(0.5, 1);
  expect(sum(pixels)).toBeCloseTo(0.5, 1);
  expect(Math.max(...points)).toBeCloseTo(0.25, 1);
  expect(Math.max(...pixels)).toBeCloseTo(0.5, 1);
  root.destroy();
});

it('draws no blended segment whose ends both have width 0, and one that narrows to it', async () => {
  const root = await gpuRoot();
  // Paint every pixel the line rasterises, whatever it covers
  const solid = lines.create(root, {
    capacity: 2,
    length: 3,
    units: 'points',
    shade: () => {
      'use gpu';
      return d.vec4f(1);
    },
  });
  solid.write(0, [-4, 2, 0, 0, 2, 0, 4, 2, 0], { width: [0, 0, 2] });
  solid.write(1, [-4, -2, 0, 4, -2, 0], { width: 0 });

  const red = await drawn(root, solid);
  // The first line's left half and the whole second line have no width
  expect(sum(column(red, 16))).toBe(0);
  expect(sum(column(red, 48))).toBeGreaterThan(0);
  root.destroy();
});

it('draws world widths that shrink with distance under perspective', async () => {
  const root = await gpuRoot();
  const view = cameras.custom(root);
  const { mat4 } = await import('wgpu-matrix');
  view.write(mat4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]), mat4.perspective(1, 1, 0.1, 100));
  const paths = lines.create(root, { capacity: 2, length: 2 });
  paths.write(0, [-1, 0.5, -4, 1, 0.5, -4], { width: 0.2 });
  paths.write(1, [-1, -0.5, -8, 1, -0.5, -8], { width: 0.2 });

  const red = await drawn(root, paths, view);
  const near = sum(column(red, 32).slice(0, 32));
  const far = sum(column(red, 32).slice(32));
  // tan(0.5) puts 1.83 units across 64 pixels at depth 1
  expect(near).toBeCloseTo((0.2 / 4) * (64 / (2 * Math.tan(0.5))), 0);
  expect(near / far).toBeCloseTo(2, 1);
  root.destroy();
});

it('keeps a segment that passes the eye, cut at the near plane, under standard and reversed depth', async () => {
  const root = await gpuRoot();
  const view = cameras.custom(root);
  const { mat4 } = await import('wgpu-matrix');
  const paths = lines.create(root, { capacity: 1, length: 2 });
  paths.write(0, [0, -0.5, 5, 0, -0.5, -20], { width: 0.05 });

  for (const projection of [
    mat4.perspective(1, 1, 0.1, 100),
    mat4.perspectiveReverseZ(1, 1, 0.1, 100),
  ]) {
    view.write(mat4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]), projection);
    const red = await drawn(root, paths, view);
    // From the bottom edge up to just below the centre, and nothing above it
    expect(sum(red)).toBeGreaterThan(20);
    expect(sum(red.slice(0, 30 * SIZE))).toBeLessThan(0.5);
  }
  root.destroy();
});

it('joins the end cut at the near plane, whatever its cap', async () => {
  const root = await gpuRoot();
  const view = cameras.custom(root);
  const { mat4 } = await import('wgpu-matrix');
  view.write(mat4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]), mat4.perspective(1, 1, 0.1, 100));
  const covered = [];
  for (const startCap of [caps.butt, caps.round, caps.arrow]) {
    const cut = lines.create(root, { capacity: 1, length: 2, startCap });
    // From behind the eye to far ahead, cut a third of the way down the screen
    cut.write(0, [0, -0.02, 1, 0, -0.02, -20], { width: 0.01 });
    covered.push(sum(await drawn(root, cut, view)));
  }
  expect(covered[0]).toBeGreaterThan(20);
  expect(covered[1]).toBeCloseTo(covered[0], 0);
  expect(covered[2]).toBeCloseTo(covered[0], 0);
  root.destroy();
});

it('caps the end of a line at a repeated point', async () => {
  const root = await gpuRoot();
  const paused = lines.create(root, {
    capacity: 1,
    length: 4,
    endCap: caps.square,
  });
  // Right to the centre, the centre again, then up
  paused.write(0, [-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], { width: 2 });

  const red = await drawn(root, paused);
  // Inside the square cap's corner below and right of the centre, at (0.86, -0.86)
  const x = Math.floor(31.5 + 0.86 * 6.4 + 0.5);
  const y = Math.floor(31.5 + 0.86 * 6.4 + 0.5);
  expect(red[y * SIZE + x]).toBeGreaterThan(0.9);
  root.destroy();
});

it('covers a right angle most with miter, then round, then bevel joins', async () => {
  const root = await gpuRoot();
  const covered = [];
  for (const join of [joins.miter, joins.round, lines.joins.bevel]) {
    const corner = lines.create(root, {
      capacity: 1,
      length: 3,
      join,
      startCap: caps.butt,
      endCap: caps.butt,
    });
    corner.write(0, [-3, -3, 0, 0, 0, 0, 3, -3, 0], { width: 2 });
    covered.push(sum(await drawn(root, corner)));
  }
  expect(covered[0]).toBeGreaterThan(covered[1] + 3);
  expect(covered[1]).toBeGreaterThan(covered[2] + 1);
  root.destroy();
});

it('covers more with butt, round, square, then arrow end caps', async () => {
  const root = await gpuRoot();
  const covered = [];
  for (const endCap of [caps.butt, caps.round, caps.square, caps.arrow]) {
    const capped = lines.create(root, {
      capacity: 1,
      length: 2,
      startCap: caps.butt,
      endCap,
    });
    capped.write(0, [-3, 0, 0, 2, 0, 0], { width: 1 });
    covered.push(sum(await drawn(root, capped)));
  }
  // A unit-wide line five units long covers 32 by 6.4 pixels; caps add to its end
  expect(covered[0]).toBeCloseTo(32 * 6.4, -1);
  for (let i = 1; i < covered.length; i++) expect(covered[i]).toBeGreaterThan(covered[i - 1] + 4);
  root.destroy();
});

it('lays a cap that reaches back along its segment at the segment depth there', async () => {
  const root = await gpuRoot();
  const view = cameras.custom(root);
  const { mat4 } = await import('wgpu-matrix');
  view.write(mat4.lookAt([0, 1.5, 0], [0, 0, -3], [0, 1, 0]), mat4.perspective(1, 1, 0.1, 100));
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { color: { format: 'rgba8unorm' }, depth: { format: 'depth24plus' } },
  );
  const into = passes
    .create(root, screen, { color: passes.color, depth: passes.depth })
    .pipe(view.inject());
  const options = { capacity: 1, length: 2, blend: 'opaque', startCap: caps.butt } as const;
  // A wide green band receding along the ground, and a red arrow lying just over it
  const ground = lines.create(root, { ...options, endCap: caps.butt });
  ground.write(0, [0, -0.03, -1, 0, -0.03, -8], { width: 3, color: [0, 1, 0, 1] });
  const arrow = lines.create(root, { ...options, endCap: caps.arrow });
  arrow.write(0, [0, 0, -1.5, 0, 0, -4.5], { width: 0.3, color: [1, 0, 0, 1] });

  const red = async (...sets: lines.Lines[]) => {
    const encoder = root.createCommandEncoder();
    into.record(
      encoder,
      sets.map((set) => into.add(set.drawing)),
    );
    encoder.submit();
    return sum(await readRed(root, root.unwrap(screen.textures.color)));
  };
  const alone = await red(arrow);
  expect(await red(ground, arrow)).toBeGreaterThan(alone * 0.95);
  into.destroy();
  screen.destroy();
  root.destroy();
});

it('closes a ring generated by a point function, with no buffer, and stops it at a count', async () => {
  const root = await gpuRoot();
  const circle = {
    capacity: 1,
    length: 48,
    point: (_line: number, i: number) => {
      'use gpu';
      const angle = (d.f32(i) / 48) * Math.PI * 2;
      return lines.ControlPoint({
        position: d.vec3f(std.cos(angle), std.sin(angle), 0) * 3,
        width: 0.5,
        color: d.vec4f(1),
        along: 0,
      });
    },
  };
  const ring = lines.create(root, { ...circle, closed: true });
  expect(ring.points).toBeUndefined();
  // Each of the ring's 8 compass points, 3 units from the centre, 19.2 pixels
  const compass = (red: Float32Array) =>
    Array.from({ length: 8 }, (_, k) => {
      const x = Math.round(31.5 + 19.2 * Math.cos((k * Math.PI) / 4));
      const y = Math.round(31.5 - 19.2 * Math.sin((k * Math.PI) / 4));
      return red[y * SIZE + x] > 0.5;
    });

  const red = await drawn(root, ring);
  expect(compass(red)).toEqual(Array(8).fill(true));
  expect(red[32 * SIZE + 32]).toBe(0);

  const arc = lines.create(root, {
    ...circle,
    count: () => {
      'use gpu';
      return d.u32(24);
    },
  });
  // The first 24 points, from angle 0 to just short of half a turn
  expect(compass(await drawn(root, arc))).toEqual([
    true,
    true,
    true,
    true,
    false,
    false,
    false,
    false,
  ]);
  root.destroy();
});

it('draws what a kernel writes, up to the count it sets', async () => {
  const root = await gpuRoot();
  const paths = lines.create(root, { capacity: 2, length: 8 });
  const trace = tgpu.computeFn({ workgroupSize: [8], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x >= 8) return;
    const x = d.f32(gid.x) - 3.5;
    paths.setPoint(
      1,
      gid.x,
      lines.ControlPoint({ position: d.vec3f(x, 2, 0), width: 1, color: d.vec4f(1), along: x }),
    );
    if (gid.x === 0) paths.setCount(1, 4);
  });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(root.pipe(paths.inject()).createComputePipeline({ compute: trace }));
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();

  const red = await drawn(root, paths);
  // Four points span x from -3.5 to -0.5, row 19 at y = 2
  expect(red[19 * SIZE + 16]).toBeGreaterThan(0.9);
  expect(red[19 * SIZE + 40]).toBe(0);
  root.destroy();
});

it('cuts dashes along the line in the units of along', async () => {
  const root = await gpuRoot();
  const dashed = lines.create(root, {
    capacity: 1,
    length: 2,
    shade: (fragment) => {
      'use gpu';
      const alpha = fragment.coverage * lines.dash(fragment, 1, 1);
      return d.vec4f(alpha, alpha, alpha, alpha);
    },
  });
  dashed.write(0, [-4, 0.03, 0, 4, 0.03, 0], { width: 1 });

  const row = Array.from((await drawn(root, dashed)).slice(32 * SIZE, 33 * SIZE));
  // On from -4 to -3, off to -2, …: half the 8 units' 51.2 pixels
  expect(sum(row)).toBeCloseTo(25.6, -1);
  expect(row[Math.round(31.5 - 3.5 * 6.4)]).toBeGreaterThan(0.9);
  expect(row[Math.round(31.5 - 2.5 * 6.4)]).toBeLessThan(0.1);
  root.destroy();
});

it('keeps the end of a dash straight across a tapering line', async () => {
  const root = await gpuRoot();
  const dashed = lines.create(root, {
    capacity: 1,
    length: 2,
    shade: (fragment) => {
      'use gpu';
      const alpha = fragment.coverage * lines.dash(fragment, 1, 1);
      return d.vec4f(alpha, alpha, alpha, alpha);
    },
  });
  dashed.write(0, [-4.47, 0, 0, 4.53, 0, 0], { width: [0.2, 3.4] });

  const red = await drawn(root, dashed);
  // A dash ends at along = 7, x = 2.53 units, where the line is 17 pixels wide
  const ends = Array.from({ length: 13 }, (_, k) => {
    const row = red.subarray((26 + k) * SIZE, (27 + k) * SIZE);
    const x = row.findIndex((value, i) => i > 44 && value < 0.5);
    return x - 0.5 + (row[x - 1] - 0.5) / (row[x - 1] - row[x]);
  });
  for (const end of ends) expect(end).toBeCloseTo(32 + 2.53 * 6.4, 0);
  expect(Math.max(...ends) - Math.min(...ends)).toBeLessThan(0.2);
  root.destroy();
});

it('hides what lies behind an opaque line, with and without multisampling', async () => {
  const root = await gpuRoot();
  const view = flat(root);
  const options = { capacity: 1, length: 2, blend: 'opaque' } as const;
  // A red line in front, at z = 0.5, drawn before a white one crossing behind it
  const front = lines.create(root, options);
  front.write(0, [-4, 0.03, 0.5, 4, 0.03, 0.5], { width: 1, color: [1, 0, 0, 1] });
  const behind = lines.create(root, options);
  behind.write(0, [0.03, -4, -0.5, 0.03, 4, -0.5], { width: 1 });

  for (const samples of [1, 4]) {
    const screen = targets.create(
      root,
      { width: SIZE, height: SIZE },
      {
        drawn: { format: 'rgba8unorm', samples },
        color: { format: 'rgba8unorm' },
        depth: { format: 'depth24plus', samples },
      },
    );
    const into = passes
      .create(
        root,
        screen,
        { drawn: passes.color, depth: passes.depth },
        samples > 1 ? { resolve: { drawn: 'color' } } : {},
      )
      .pipe(view.inject());
    const encoder = root.createCommandEncoder();
    into.record(encoder, [into.add(front.drawing), into.add(behind.drawing)]);
    encoder.submit();
    const shown = samples > 1 ? screen.textures.color : screen.textures.drawn;
    const red = await readRed(root, root.unwrap(shown));
    // Where they cross, the front line's green and blue stay 0, so only red shows; read green
    const green = await readChannel(root, root.unwrap(shown), 1);
    expect(red[32 * SIZE + 32]).toBeGreaterThan(0.95);
    expect(green[32 * SIZE + 32]).toBeLessThan(0.05);
    expect(green[10 * SIZE + 32]).toBeGreaterThan(0.95);
    into.destroy();
    screen.destroy();
  }
  root.destroy();
});

it('draws an opaque line into depth alone at its own width, and a blended one not at all', async () => {
  const root = await gpuRoot();
  const view = flat(root);
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { depth: { format: 'depth32float' } },
  );
  const shadow = passes.create(root, screen, { depth: passes.depth }).pipe(view.inject());
  const written = async (blend: lines.Blend) => {
    // Four points wide about the boundary between rows 31 and 32
    const set = lines.create(root, { capacity: 1, length: 2, units: 'points', blend });
    set.write(0, [-4, 0, 0, 4, 0, 0], { width: 4 });
    const encoder = root.createCommandEncoder();
    shadow.record(encoder, shadow.add(set.drawing));
    encoder.submit();
    const depth = await readDepth(root, root.unwrap(screen.textures.depth));
    return column(depth, 32).filter((value) => value < 1).length;
  };

  expect(await written('opaque')).toBe(4);
  expect(await written('over')).toBe(0);
  shadow.destroy();
  screen.destroy();
  root.destroy();
});

it('joins an end whose neighbour lies behind the eye, so butt caps leave no notch', async () => {
  const root = await gpuRoot();
  const view = cameras.custom(root);
  const { mat4 } = await import('wgpu-matrix');
  view.write(mat4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]), mat4.perspective(1, 1, 0.1, 100));
  const bent = lines.create(root, {
    capacity: 1,
    length: 3,
    units: 'points',
    startCap: caps.butt,
    endCap: caps.butt,
  });
  // Up the screen from behind the eye to the centre, then right: a round join 8 pixels across
  bent.write(0, [0, -1, 2, 0, 0, -4, 1.5, 0, -4], { width: 16 });

  const red = await drawn(root, bent, view);
  // Above and left of the corner, inside the join's outer half that the second segment adds
  expect(red[25 * SIZE + 28]).toBeGreaterThan(0.9);
  root.destroy();
});

it('fades a butt end over the pixel it crosses', async () => {
  const root = await gpuRoot();
  const butted = lines.create(root, {
    capacity: 1,
    length: 2,
    units: 'points',
    startCap: caps.butt,
    endCap: caps.butt,
  });
  // Ends at x = 34.37 pixels, a third of the way across pixel 34
  const end = ((34.37 - 32) / 64) * 10;
  butted.write(0, [-2, 0.03, 0, end, 0.03, 0], { width: 4 });

  const row = (await drawn(root, butted)).slice(31 * SIZE, 32 * SIZE);
  expect(row[33]).toBeGreaterThan(0.95);
  expect(row[34]).toBeCloseTo(0.37, 1);
  expect(row[35]).toBeLessThan(0.02);
  root.destroy();
});

it('draws a thin line in points continuously at a shallow angle, with and without multisampling', async () => {
  const root = await gpuRoot();
  const view = flat(root);
  const plain = lines.create(root, { capacity: 1, length: 2, units: 'points' });
  // A profile across the line, which each pixel averages over the part of it the line covers
  const profiled = lines.create(root, {
    capacity: 1,
    length: 2,
    units: 'points',
    shade: (fragment) => {
      'use gpu';
      const alpha = (1 - fragment.across * fragment.across) * fragment.coverage;
      return d.vec4f(alpha, alpha, alpha, alpha);
    },
  });
  // Rising one pixel every eight, so the line crosses rows at every offset
  for (const set of [plain, profiled]) set.write(0, [-5, -0.4, 0, 5, 0.85, 0], { width: 1.5 });

  const size = { width: SIZE, height: SIZE };
  const single = targets.create(root, size, { drawn: { format: 'rgba8unorm' } });
  const once = passes.create(root, single, { drawn: passes.color }).pipe(view.inject());
  const multi = targets.create(root, size, {
    drawn: { format: 'rgba8unorm', samples: 4 },
    color: { format: 'rgba8unorm' },
  });
  const resolved = passes
    .create(root, multi, { drawn: passes.color }, { resolve: { drawn: 'color' } })
    .pipe(view.inject());
  const check = async (into: typeof once, shown: GPUTexture) => {
    for (const [set, share] of [
      [plain, 1.5],
      [profiled, 1],
    ] as const) {
      const encoder = root.createCommandEncoder();
      into.record(encoder, into.add(set.drawing));
      encoder.submit();
      const red = await readRed(root, shown);
      const columns = Array.from({ length: 48 }, (_, k) => column(red, k + 8));
      const peaks = columns.map((each) => Math.max(...each));
      // Every column holds the same light, and none of its pixels falls far below the brightest
      for (const each of columns) expect(sum(each)).toBeCloseTo(share, 1);
      expect(Math.min(...peaks)).toBeGreaterThan(Math.max(...peaks) * 0.55);
    }
  };

  await check(once, root.unwrap(single.textures.drawn));
  await check(resolved, root.unwrap(multi.textures.color));
  for (const each of [once, resolved, single, multi]) each.destroy();
  root.destroy();
});
