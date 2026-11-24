import { lowerReferences } from './references.ts';
import { layoutOf } from './memory.ts';
import { NodeTypeCatalog as NODE, type Expression, type Statement } from 'tinyest';
import { d, std } from 'typegpu';
import {
  WgslGenerator,
  snip,
  type BinaryOperator,
  type ShaderGenerator,
  type ConstantDefinitionOptions,
  type FunctionDefinitionOptions,
  type ResolvedSnippet,
  type ResolvedStatement,
  type Snippet,
  type VariableDefinitionOptions,
} from 'typegpu/~internal';

function unsupported(name: string): never {
  throw new Error(`@typegpu/kit-gl CPU: unsupported ${name}`);
}

function scalar(schema: d.BaseData | symbol): string {
  if (typeof schema === 'symbol') return unsupported('unknown type');
  if (schema.type === 'decorated') return scalar((schema as d.Decorated).inner);
  return schema.type;
}

export class CpuGenerator extends WgslGenerator {
  override languageKey = 'javascript';
  #functions = new Set<string>();
  #generators = new Set<string>();
  #types = new Map<d.BaseData, string>();

  protected override _expression(expression: Expression): Snippet {
    // Preserve integer overflow for unary negation before subsequent conversions.
    // eslint-disable-next-line no-underscore-dangle
    const result = super._expression(expression);
    if (
      Array.isArray(expression) &&
      expression[0] === NODE.unaryExpr &&
      scalar(result.dataType) === 'i32'
    ) {
      return snip(
        `((${this.ctx.resolveSnippet(result).value}) | 0)`,
        result.dataType as d.BaseData,
        result.origin,
        result.possibleSideEffects,
      );
    }
    return result;
  }

  protected override _statement(statement: Statement): ResolvedStatement {
    if (Array.isArray(statement) && statement[0] === NODE.postUpdate) {
      // eslint-disable-next-line no-underscore-dangle
      const value = this._expression(statement[2]);
      return {
        code: `${this.emitBinaryOp(value, statement[1] === '++' ? '+=' : '-=', snip('1', value.dataType as d.BaseData, 'constant'))};`,
        definesInNearestScope: false,
      };
    }
    // eslint-disable-next-line no-underscore-dangle
    return super._statement(statement);
  }

  protected override _emitSwitchStatement(
    discriminant: Snippet,
    groups: [Snippet[], ResolvedStatement[]][],
  ): string {
    return `switch (${this.ctx.resolveSnippet(discriminant).value}) {${groups.map(([tests, statements]) => `${tests.map((test) => (test.value === 'default' ? 'default:' : `case ${this.ctx.resolveSnippet(test).value}:`)).join('')} {${statements.map((statement) => statement.code).join('\n')} break;}`).join('\n')}}`;
  }

  protected override _aliasConstStatement(
    rawId: string,
    _node: Expression,
    value: Snippet,
  ): ResolvedStatement {
    const type = scalar(value.dataType);
    if (type === 'ptr') return unsupported('implicit pointer alias');
    const name = this.ctx.makeUniqueIdentifier(rawId, 'block');
    this.ctx.defineVariable(rawId, snip(name, value.dataType as d.BaseData, value.origin, false));
    return {
      code: `const ${name} = ${this.ctx.resolveSnippet(value).value};`,
      definesInNearestScope: true,
    };
  }

  override call(...parameters: Parameters<ShaderGenerator['call']>): Snippet {
    const [callee, args] = parameters;
    const name = String(callee);
    if (name.startsWith('atomic')) {
      const target = lowerReferences(this.ctx.resolveSnippet(args[0]!).value);
      const indexed = /^(.*)\[([^[\]]+)\]$/.exec(target);
      const field = /^(.*)\.([A-Za-z_]\w*)$/.exec(target);
      const object = indexed?.[1] ?? field?.[1];
      const key = indexed?.[2] ?? (field ? JSON.stringify(field[2]) : undefined);
      if (!object || key === undefined) unsupported(`atomic reference ${target}`);
      let schema = args[0]!.dataType as d.BaseData;
      if (d.isPtr(schema)) schema = schema.inner;
      if (!d.isAtomic(schema)) unsupported(`atomic type ${schema.type}`);
      const inner = schema.inner;
      const values = args.slice(1).map((arg) => this.ctx.resolveSnippet(arg).value);
      const operations: Record<string, string> = {
        atomicLoad: 'return old;',
        atomicStore: 'a[k]=v;',
        atomicAdd: 'a[k]=old+v;return old;',
        atomicSub: 'a[k]=old-v;return old;',
        atomicMin: 'a[k]=Math.min(old,v);return old;',
        atomicMax: 'a[k]=Math.max(old,v);return old;',
        atomicAnd: 'a[k]=old&v;return old;',
        atomicOr: 'a[k]=old|v;return old;',
        atomicXor: 'a[k]=old^v;return old;',
        atomicExchange: 'a[k]=v;return old;',
      };
      let operation = operations[name];
      if (operation)
        operation = operation.replace(
          /a\[k\]=([^;]+);/g,
          `a[k]=(($1) ${inner.type === 'u32' ? '>>>0' : '|0'});`,
        );
      if (!operation) unsupported(`builtin ${name}`);
      return snip(
        `((a,k,v)=>{const old=a[k];${operation}})(${object},${key},${values[0] ?? '0'})`,
        name === 'atomicStore' ? d.Void : inner,
        'runtime',
        true,
      );
    }
    const result = super.call(...parameters);
    if (
      result.origin === 'constant' ||
      [
        'comptime',
        'typeOf',
        'isKnownAtComptime',
        'unroll',
        'range',
        'add',
        'sub',
        'mul',
        'div',
        'mod',
        'intdiv',
      ].includes(name)
    )
      return result;
    if (name === 'copy') return this.typeInstantiation(result.dataType as d.BaseData, args);
    if (['workgroupBarrier', 'storageBarrier', 'textureBarrier'].includes(name)) return result;
    if (Object.values(std).includes(callee as never)) {
      return snip(
        this.emitCall(name, [], args),
        result.dataType as d.BaseData,
        result.origin,
        result.possibleSideEffects,
      );
    }
    return result;
  }

  override numericLiteral(value: number, schema: d.BaseData): ResolvedSnippet {
    return snip(String(value), schema, 'constant', false);
  }

  override emitTypeAnnotation(schema: d.BaseData): string {
    const type = scalar(schema);
    if (type === 'ptr') return 'cpuRef';
    if (['u32', 'i32', 'f32', 'bool', 'void', 'abstractInt', 'abstractFloat'].includes(type)) {
      return type;
    }
    const known = this.#types.get(schema);
    if (known) return known;
    const layout = layoutOf(schema);
    const name = this.ctx.makeUniqueIdentifier('cpuType', 'global');
    this.#types.set(schema, name);
    this.ctx.addDeclaration(
      `function ${name}(...args) { return $construct(${JSON.stringify(layout)}, ...args); }`,
      name,
    );
    return name;
  }

  override typeInstantiation(schema: d.BaseData, args: readonly Snippet[]): ResolvedSnippet {
    const type = this.emitTypeAnnotation(schema);
    const value = args.length ? this.ctx.resolveSnippet(args[0]!).value : '0';
    if (type.startsWith('cpuType'))
      return snip(
        `${type}(${args.map((arg) => this.ctx.resolveSnippet(arg).value).join(',')})`,
        schema,
        'runtime',
        args.some((arg) => arg.possibleSideEffects),
      );
    if (args.length > 1) unsupported(`constructor ${type}`);
    const code =
      type === 'u32'
        ? `((${value}) >>> 0)`
        : type === 'i32'
          ? `((${value}) | 0)`
          : type === 'bool'
            ? `Boolean(${value})`
            : `Number(${value})`;
    return snip(
      code,
      schema,
      'runtime',
      args.some((arg) => arg.possibleSideEffects),
    );
  }

  protected override _emitVarDecl(
    _keyword: string,
    name: string,
    schema: d.BaseData,
    rhs: string,
  ): string {
    const type = this.emitTypeAnnotation(schema);
    return `let ${name} = ${type.startsWith('cpuType') ? `${type}(${rhs})` : rhs};`;
  }

  override declareGlobalConst(options: ConstantDefinitionOptions): ResolvedSnippet {
    this.emitTypeAnnotation(options.dataType);
    this.ctx.addDeclaration(
      `const ${options.id} = ${this.ctx.resolveSnippet(options.init).value};`,
      options.id,
    );
    return snip(options.id, options.dataType, 'constant-immutable-def');
  }

  override declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    const { dataType, id, scope, group, binding } = options;
    if (scope === 'private' || scope === 'workgroup') {
      const ctor = this.emitTypeAnnotation(dataType);
      const initial = options.init
        ? this.ctx.resolveSnippet(options.init).value
        : ctor.startsWith('cpuType')
          ? `${ctor}()`
          : '0';
      const scalarType = ['u32', 'i32', 'f32', 'bool'].includes(scalar(dataType));
      const expression =
        scope === 'private'
          ? initial
          : `$shared(${JSON.stringify(id)},()=>(${scalarType ? `{value:${initial}}` : initial}))`;
      this.ctx.addDeclaration(`//#scope ${scope}\nlet ${id} = ${expression};`, id);
      return snip(scope === 'workgroup' && scalarType ? `${id}.value` : id, dataType, scope);
    }
    if (!['uniform', 'readonly', 'mutable'].includes(scope)) unsupported(`${scope} variable ${id}`);
    const array = dataType.type === 'array' ? (dataType as d.WgslArray) : undefined;
    const element = array?.elementType ?? dataType;
    const type = scalar(d.isAtomic(element) ? element.inner : element);
    if (!['u32', 'i32', 'f32'].includes(type)) {
      const layout = layoutOf(dataType);
      this.ctx.addDeclaration(
        `//#binding ${JSON.stringify({ group, binding, kind: 'buffer', reads: true, writes: scope === 'mutable' })}\nconst ${id} = $memory(${group}, ${binding}, ${JSON.stringify(layout)});`,
        id,
      );
      return snip(id, dataType, scope);
    }
    const ctor = type === 'u32' ? 'Uint32Array' : type === 'i32' ? 'Int32Array' : 'Float32Array';
    this.ctx.addDeclaration(
      `//#binding ${JSON.stringify({ group, binding, kind: 'buffer', reads: true, writes: scope === 'mutable' })}\nconst ${id} = $buffer(${group}, ${binding}, ${ctor});`,
      id,
    );
    return snip(array ? id : `${id}[0]`, dataType, scope);
  }

  override functionDefinition(options: FunctionDefinitionOptions): string {
    if (options.functionType !== 'normal' && options.functionType !== 'compute')
      unsupported(`${options.functionType} entry point`);
    this.#functions.add(options.name);
    // The protected AST emitter is the extension hook TypeGPU exposes.
    // eslint-disable-next-line no-underscore-dangle
    let body = lowerReferences(this._block(options.body, false).code);
    body = body.replace(/\b(?:workgroupBarrier|storageBarrier|textureBarrier)\(\)/g, '(yield)');
    for (const name of this.#generators) {
      const pattern = new RegExp(`\\b${name}\\(`, 'g');
      body = body.replace(pattern, `(yield* ${name}(`);
      let offset = body.indexOf(`(yield* ${name}(`);
      while (offset !== -1) {
        let end = offset + `(yield* ${name}(`.length;
        let depth = 1;
        for (; end < body.length && depth; end++) {
          if (body[end] === '(') depth++;
          if (body[end] === ')') depth--;
        }
        body = body.slice(0, end) + ')' + body.slice(end);
        offset = body.indexOf(`(yield* ${name}(`, end + 1);
      }
    }
    const generator = body.includes('(yield');
    if (generator) this.#generators.add(options.name);
    const returnType = options.determineReturnType();
    const returns = this.emitTypeAnnotation(returnType);
    if (returns.startsWith('cpuType'))
      body = body.replace(/return ([^;]+);/g, `return ${returns}($1);`);
    const args = options.args.filter((arg) => arg.used || options.functionType === 'normal');
    if (options.functionType === 'normal') {
      const copies = args
        .map((arg) => {
          const type = this.emitTypeAnnotation(arg.decoratedType);
          return type.startsWith('cpuType') ? `${arg.name} = ${type}(${arg.name});` : '';
        })
        .join('');
      body = body.replace('{', `{${copies}`);
      return `function${generator ? '*' : ''} ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}`;
    }
    const inputs = args.map((arg) => {
      const schema = arg.decoratedType as d.Decorated;
      const builtin = (schema.attribs as { type: string; params: string[] }[] | undefined)?.find(
        (attr) => attr.type === '@builtin',
      );
      if (
        !builtin ||
        ![
          'global_invocation_id',
          'local_invocation_id',
          'local_invocation_index',
          'workgroup_id',
          'num_workgroups',
        ].includes(builtin.params[0]!)
      )
        unsupported(`compute input ${arg.name}`);
      return builtin.params[0]!;
    });
    const size = [1, 1, 1];
    options.workgroupSize?.forEach((value, index) => {
      size[index] = value;
    });
    return `//#kit-gl compute\n//#entry ${JSON.stringify({ name: options.name, size, inputs, generator })}\nfunction${generator ? '*' : ''} ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}`;
  }

  override emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
    const type = scalar(lhs.dataType);
    const rhsType = scalar(rhs.dataType);
    if (type.startsWith('vec') || rhsType.startsWith('vec')) {
      if (['==', '!=', '<', '>', '<=', '>='].includes(op)) unsupported('vector comparison');
      const schema = (type.startsWith('vec') ? lhs.dataType : rhs.dataType) as d.BaseData;
      const vectorType = scalar(schema);
      const size = Number(vectorType[3]);
      const primitive = vectorType.endsWith('u') ? d.u32 : vectorType.endsWith('i') ? d.i32 : d.f32;
      const left = this.ctx.resolveSnippet(lhs).value;
      const right = this.ctx.resolveSnippet(rhs).value;
      const ctor = this.emitTypeAnnotation(schema);
      if (op === '=') return `${left} = ${ctor}(${right})`;
      const assignment = op.endsWith('=') && !['==', '!=', '<=', '>='].includes(op);
      const operation = (assignment ? op.slice(0, -1) : op) as BinaryOperator;
      const components = Array.from({ length: size }, (_, i) =>
        this.emitBinaryOp(
          snip(type.startsWith('vec') ? `a[${i}]` : 'a', primitive, 'runtime'),
          operation,
          snip(rhsType.startsWith('vec') ? `b[${i}]` : 'b', primitive, 'runtime'),
        ),
      );
      const code = `((a,b)=>${ctor}(${components.join(',')}))(${left},${right})`;
      return assignment ? `${left} = ${code}` : code;
    }
    if (op === '=' && ['struct', 'array'].includes(type))
      return `${this.ctx.resolveSnippet(lhs).value} = ${this.emitTypeAnnotation(lhs.dataType as d.BaseData)}(${this.ctx.resolveSnippet(rhs).value})`;

    if (!['u32', 'i32', 'f32', 'bool', 'abstractInt', 'abstractFloat'].includes(type))
      unsupported(`${op} on ${type}`);
    const left = this.ctx.resolveSnippet(lhs).value;
    const right = this.ctx.resolveSnippet(rhs).value;
    if (op === '=') return `${left} = ${right}`;
    const assignment = op.endsWith('=') && !['==', '!=', '<=', '>='].includes(op);
    const operator = assignment ? op.slice(0, -1) : op;
    const integer = type === 'u32' || type === 'i32';
    let result =
      integer && operator === '*'
        ? `Math.imul(${left}, ${right})`
        : integer && ['/', '%'].includes(operator)
          ? `(${left} ${operator} ((${right}) || 1))`
          : `(${left} ${operator === '>>' && type === 'u32' ? '>>>' : operator} ${right})`;
    if (integer && !['<', '>', '<=', '>=', '==', '!=', '&&', '||'].includes(operator))
      result = `(${result} ${type === 'u32' ? '>>> 0' : '| 0'})`;
    return assignment ? `${left} = ${result}` : result;
  }

  override emitCall(name: string, templates: readonly Snippet[], args: readonly Snippet[]): string {
    if (name === 'bitcast') {
      const target = this.ctx.resolveSnippet(templates[0]!).value;
      const source = scalar(args[0]!.dataType);
      if (!['u32', 'i32', 'f32'].includes(target) || !['u32', 'i32', 'f32'].includes(source))
        unsupported(`bitcast ${source} to ${target}`);
      return `$bits.${source}_${target}(${this.ctx.resolveSnippet(args[0]!).value})`;
    }
    if (templates.length) unsupported(`template call ${name}`);
    const values = args.map((arg) => this.ctx.resolveSnippet(arg).value);

    const vectorArg = args.find((arg) => scalar(arg.dataType).startsWith('vec'));
    if (vectorArg) {
      const schema = vectorArg.dataType as d.BaseData;
      const count = Number(schema.type[3]);
      const ctor = this.emitTypeAnnotation(schema);
      const letters = args.map((_, i) => `a${i}`);
      const component = (index: number, i: number) =>
        scalar(args[index]!.dataType).startsWith('vec')
          ? `${letters[index]}[${i}]`
          : letters[index]!;
      const dot = Array.from(
        { length: count },
        (_, i) => `${component(0, i)}*${component(name === 'dot' ? 1 : 0, i)}`,
      ).join('+');
      let body: string;
      if (name === 'dot' || name === 'length')
        body = name === 'dot' ? `(${dot})` : `Math.sqrt(${dot})`;
      else if (name === 'normalize')
        body = `${ctor}(${Array.from({ length: count }, (_, i) => `${component(0, i)}/Math.sqrt(${dot})`).join(',')})`;
      else if (name === 'cross')
        body = `${ctor}(a0[1]*a1[2]-a0[2]*a1[1],a0[2]*a1[0]-a0[0]*a1[2],a0[0]*a1[1]-a0[1]*a1[0])`;
      else
        body = `${ctor}(${Array.from({ length: count }, (_, i) =>
          this.emitCall(
            name,
            [],
            args.map((arg, index) =>
              snip(
                component(index, i),
                scalar(arg.dataType).startsWith('vec') ? d.f32 : (arg.dataType as d.BaseData),
                'runtime',
              ),
            ),
          ),
        ).join(',')})`;
      return `((${letters.join(',')})=>${body})(${values.join(',')})`;
    }
    if (name === 'countOneBits')
      return `((a)=>{a=a-((a>>>1)&0x55555555);a=(a&0x33333333)+((a>>>2)&0x33333333);return Math.imul((a+(a>>>4))&0x0f0f0f0f,0x01010101)>>>24;})(${values[0]})`;
    const expressions: Record<string, string> = {
      clamp: 'Math.min(Math.max(a,b),c)',
      mix: 'a*(1-c)+b*c',
      fract: 'a-Math.floor(a)',
      inverseSqrt: '1/Math.sqrt(a)',
      saturate: 'Math.min(Math.max(a,0),1)',
      step: 'b<a?0:1',
      fma: 'a*b+c',
      radians: 'a*Math.PI/180',
      degrees: 'a*180/Math.PI',
      countLeadingZeros: 'Math.clz32(a)',
      countTrailingZeros: 'a===0?32:31-Math.clz32((a&-a)>>>0)',
      firstTrailingBit: 'a===0?4294967295:31-Math.clz32((a&-a)>>>0)',
    };
    if (expressions[name]) return `((a,b,c)=>(${expressions[name]}))(${values.join(',')})`;
    for (const arg of args) {
      if (
        !['u32', 'i32', 'f32', 'bool', 'abstractInt', 'abstractFloat'].includes(
          scalar(arg.dataType),
        )
      )
        unsupported(`${name} argument ${scalar(arg.dataType)}`);
    }
    if (this.#functions.has(name)) return `${name}(${values.join(',')})`;
    if (
      [
        'abs',
        'ceil',
        'floor',
        'trunc',
        'round',
        'min',
        'max',
        'sin',
        'cos',
        'tan',
        'asin',
        'acos',
        'atan',
        'atan2',
        'exp',
        'exp2',
        'log',
        'log2',
        'sqrt',
        'pow',
        'sign',
      ].includes(name)
    ) {
      if (name === 'exp2') return `(2 ** (${values[0]}))`;
      if (name === 'round') unsupported('round (ties-to-even)');
      return `Math.${name}(${values.join(',')})`;
    }
    if (name === 'select') return `(${values[2]} ? ${values[1]} : ${values[0]})`;
    return unsupported(`builtin ${name}`);
  }
}
