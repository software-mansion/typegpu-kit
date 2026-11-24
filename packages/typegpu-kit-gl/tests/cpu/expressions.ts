import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import typegpuPlugin from 'unplugin-typegpu/rollup';

const primitives = [
  'std.cross(a,b)',
  'd.vec3f(std.dot(a,b))',
  'std.mix(a,b,0.375)',
  'std.select(a,b,std.lt(a,b))',
  'std.normalize(std.select(a,d.vec3f(1),std.eq(std.dot(a,a),0)))',
  'std.reflect(a,std.normalize(b))',
  'std.refract(a,std.normalize(b),0.25)',
  'std.min(a,b)',
  'std.max(a,b)',
  'std.clamp(a,d.vec3f(-2),d.vec3f(2))',
  'd.vec3f(std.length(a))',
  'd.vec3f(std.distance(a,b))',
  'std.abs(a)',
  'std.sign(a)',
  'std.floor(a)',
  'std.fract(a)',
  'std.mod(a,b)',
  '(m*a)',
  '((m*n)*a)',
  '(-a)',
  'a.zyx',
  'd.vec3f(a.xy,b.z)',
];

export interface ExpressionCase {
  source: string;
  code: string;
}
const expression = (source: string): ExpressionCase => ({ source, code: `d.vec3f(${source})` });
const nonzero = (source: string) =>
  `std.select((${source}),d.vec3f(1),std.eq((${source}),d.vec3f(0)))`;
export const expressionCases: ExpressionCase[] = primitives.map(expression);
for (const source of primitives) {
  for (const operator of ['+', '-', '*', '/', '%']) {
    expressionCases.push(expression(`(${source}) ${operator} b`));
    expressionCases.push(
      expression(`b ${operator} (${['/', '%'].includes(operator) ? nonzero(source) : source})`),
    );
    expressionCases.push({
      source: `let value=b; value ${operator}= (${source});`,
      code: `{let value=d.vec3f(b);value ${operator}= (${['/', '%'].includes(operator) ? nonzero(source) : source});RESULT=d.vec3f(value);}`,
    });
  }
  for (const comparison of ['eq', 'ne', 'lt', 'le', 'gt', 'ge']) {
    expressionCases.push(
      expression(`std.select(d.vec3f(0),d.vec3f(1),std.${comparison}((${source}),b))`),
    );
    expressionCases.push(
      expression(`std.select(d.vec3f(0),d.vec3f(1),std.${comparison}(b,(${source})))`),
    );
  }
  for (const outer of primitives)
    expressionCases.push(expression(outer.replace(/\ba\b/g, `(${source})`)));
}
for (const operator of ['+', '-', '*'])
  for (const [left, right] of [
    ['m*n', 'm'],
    ['m', 'm*n'],
  ]) {
    for (let column = 0; column < 3; column++)
      expressionCases.push(expression(`((${left}) ${operator} (${right})).columns[${column}]`));
  }

export const inputPairs = [
  d.vec3f(2.25, -1.5, 0),
  d.vec3f(1.25, 0.75, -0.5),
  d.vec3f(-0.25, 0, 3.5),
  d.vec3f(-0.5, 1.5, 0.75),
  d.vec3f(0.75, 0.5, -2),
  d.vec3f(2, -0.75, 1.25),
  d.vec3f(-3, 2.5, 1),
  d.vec3f(0.5, 1.25, -1.5),
];

export async function expressionValues(root: TgpuRoot): Promise<number[]> {
  const input = root.createReadonly(d.arrayOf(d.vec3f, inputPairs.length), inputPairs);
  const matrices = root.createReadonly(d.arrayOf(d.mat3x3f, 2), [
    d.mat3x3f(1, 0.5, 0, -0.25, 2, 0.75, 0.5, 0, -1),
    d.mat3x3f(0.5, 1, -0.5, 0.25, -1, 0, 1, 0.5, 2),
  ]);
  const output = root.createMutable(d.arrayOf(d.vec3f, expressionCases.length * 4));
  const plugin = typegpuPlugin({ autoNamingEnabled: false }) as unknown as {
    transform: { handler(source: string, id: string): { code: string } };
  };
  // Transform generated source with the same plugin as handwritten shaders. Each composition
  // remains one AST expression, so helper boundaries cannot hide scalarization errors.
  for (let offset = 0; offset < expressionCases.length; offset += 128) {
    const selected = expressionCases.slice(offset, offset + 128);
    const statements = selected
      .map((item, index) => {
        const target = `output.$[gid.x * ${expressionCases.length} + ${offset + index}]`;
        return item.code.startsWith('{')
          ? item.code.replace('RESULT', target)
          : `${target}=${item.code};`;
      })
      .join('\n');
    const source = `const kernel=({gid})=>{'use gpu';const a=input.$[gid.x*2];const b=input.$[gid.x*2+1];const m=matrices.$[0];const n=matrices.$[1];${statements}};`;
    const transformed = plugin.transform.handler(source, 'cpu-expression-fixture.js');
    const callback = new Function(
      'd',
      'std',
      'input',
      'matrices',
      'output',
      `${transformed.code}\nreturn kernel;`,
    )(d, std, input, matrices, output) as (input: { gid: d.v3u }) => undefined;
    const kernel = tgpu.computeFn({
      workgroupSize: [4],
      in: { gid: d.builtin.globalInvocationId },
    })(callback);
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  }
  return (await output.read()).flatMap((value) => [value.x, value.y, value.z]);
}
