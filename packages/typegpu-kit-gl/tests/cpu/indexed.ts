import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export const indexedCases = [
  'vector indexed store',
  'vector compound store',
  'matrix column store and copy',
  'matrix component store',
  'struct vector field store',
  'array vector component store',
  'constant index control',
  'vector alias store',
  'vector component increment',
  'index and RHS evaluated once',
  'matrix column alias store',
] as const;

export async function indexedValues(root: TgpuRoot): Promise<number[]> {
  const Record = d.struct({ vector: d.vec4u });
  const Vectors = d.arrayOf(d.vec2u, 2);
  const output = root.createMutable(d.arrayOf(d.vec4f, indexedCases.length * 4));
  const indexCalls = tgpu.privateVar(d.u32);
  const valueCalls = tgpu.privateVar(d.u32);
  const indexOf = (index: number) => {
    'use gpu';
    indexCalls.$ += 1;
    return d.u32(index);
  };
  const nextValue = () => {
    'use gpu';
    valueCalls.$ += 1;
    return d.u32(23);
  };
  const kernel = tgpu.computeFn({
    workgroupSize: [4],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const index = gid.x;
    const column = std.intdiv(index, 2);
    const row = index % 2;
    const base = index * indexedCases.length;

    let vector = d.vec4u(1, 2, 3, 4);
    vector[index] = 7;
    output.$[base] = d.vec4f(vector);

    let compound = d.vec4u(1, 2, 3, 4);
    compound[index] += 1;
    output.$[base + 1] = d.vec4f(compound);

    let matrix = d.mat2x2f(1, 2, 3, 4);
    let value = d.vec2f(7, 8);
    // The matrix columns type is readonly, although WGSL allows a column assignment.
    (matrix.columns as [d.v2f, d.v2f])[column] = d.vec2f(value);
    value[0] = 99;
    output.$[base + 2] = d.vec4f(matrix.columns[0], matrix.columns[1]);

    let elements = d.mat2x2f(1, 2, 3, 4);
    elements.columns[column][row] = 9;
    output.$[base + 3] = d.vec4f(elements.columns[0], elements.columns[1]);

    let record = Record({ vector: d.vec4u(1, 2, 3, 4) });
    record.vector[index] = 11;
    output.$[base + 4] = d.vec4f(record.vector);

    let vectors = Vectors([d.vec2u(1, 2), d.vec2u(3, 4)]);
    vectors[column][row] = 13;
    output.$[base + 5] = d.vec4f(d.vec2f(vectors[0]), d.vec2f(vectors[1]));

    let constant = d.vec4u(1, 2, 3, 4);
    constant[2] = 17;
    output.$[base + 6] = d.vec4f(constant);

    let original = d.vec4u(1, 2, 3, 4);
    const alias = original;
    alias[index] = 19;
    output.$[base + 7] = d.vec4f(original);

    let incremented = d.vec4u(1, 2, 3, 4);
    incremented[index]++;
    output.$[base + 8] = d.vec4f(incremented);

    let evaluated = d.vec4u(1, 2, 3, 4);
    evaluated[indexOf(index)] += nextValue();
    output.$[base + 9] = d.vec4f(
      d.f32(evaluated[index]),
      d.f32(indexCalls.$),
      d.f32(valueCalls.$),
      d.f32(evaluated[(index + 1) % 4]),
    );

    let aliasedMatrix = d.mat2x2f(1, 2, 3, 4);
    const selected = aliasedMatrix.columns[column];
    selected[row] = 29;
    output.$[base + 10] = d.vec4f(aliasedMatrix.columns[0], aliasedMatrix.columns[1]);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((value) => [value.x, value.y, value.z, value.w]);
}

const replaced = (at: number, value: number) =>
  [1, 2, 3, 4].map((component, i) => (i === at ? value : component));

export function expectedIndexedValues(): number[] {
  return Array.from({ length: 4 }, (_, index) => {
    const matrix = [1, 2, 3, 4];
    matrix.splice(Math.floor(index / 2) * 2, 2, 7, 8);
    return [
      replaced(index, 7),
      replaced(index, index + 2),
      matrix,
      replaced(index, 9),
      replaced(index, 11),
      replaced(index, 13),
      replaced(2, 17),
      replaced(index, 19),
      replaced(index, index + 2),
      [index + 24, 1, 1, ((index + 1) % 4) + 1],
      replaced(index, 29),
    ].flat();
  }).flat();
}
