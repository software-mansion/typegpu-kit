import { readAddress, writeAddress, type Address } from './address.ts';
import { addressParts, lowerReferences } from './references.ts';
import { layoutOf } from './memory.ts';
import { NodeTypeCatalog as NODE, type Expression, type Statement, type Return } from 'tinyest';
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
  #privateResets: string[] = [];
  #addresses = new Map<string, Address>();
  #globalAddresses = new Map<string, Address>();
  #vectors = new Map<string, string[]>();
  #components(code: string): string[] | undefined {
    return this.#vectors.get(code.replace(/^\((\w+)\)$/, '$1'));
  }
  #vector(_schema: d.BaseData, components: string[]): string {
    const name = this.ctx.makeUniqueIdentifier('cpuVector', 'global');
    this.#vectors.set(name, components);
    return name;
  }
  #parts(snippet: Snippet): string[] | undefined {
    const code = this.ctx.resolveSnippet(snippet).value;
    const components = this.#components(code);
    if (components) return components;
    const address = this.#addresses.get(code);
    if (address?.layout.type.startsWith('vec'))
      return Array.from({ length: address.layout.count! }, (_, i) =>
        readAddress({
          layout: address.layout.element!,
          base: address.base,
          offset: `(${address.offset})+${i * 4}`,
        }),
      );
    return undefined;
  }

  #value(snippet: Snippet): string {
    const code = this.ctx.resolveSnippet(snippet).value;
    const components = this.#components(code);
    if (components) return `$vector([${components.join(',')}])`;
    const address = this.#addresses.get(code);
    return address ? readAddress(address) : code;
  }
  #generators = new Set<string>();
  #types = new Map<d.BaseData, string>();

  protected override _expression(expression: Expression): Snippet {
    // Preserve integer overflow for unary negation before subsequent conversions.
    // eslint-disable-next-line no-underscore-dangle
    const result = super._expression(expression);
    if (
      typeof result.dataType !== 'symbol' &&
      result.dataType.type !== 'ptr' &&
      result.origin !== 'constant'
    ) {
      const code = this.ctx.resolveSnippet(result).value;
      const components = this.#components(code);
      if (components)
        return snip(
          this.#vector(result.dataType, components),
          result.dataType,
          result.origin,
          result.possibleSideEffects,
        );
      const address = this.#addresses.get(code);
      if (address && !/^cpuAddress(?:_\d+)?$/.test(code)) {
        const name = this.ctx.makeUniqueIdentifier('cpuAddress', 'global');
        this.#addresses.set(name, address);
        return snip(name, result.dataType, result.origin, result.possibleSideEffects);
      }
    }
    if (Array.isArray(expression) && expression[0] === NODE.objectExpr) {
      let code = this.ctx.resolveSnippet(result).value;
      if (typeof result.dataType !== 'symbol' && result.dataType.type === 'struct') {
        const type = this.emitTypeAnnotation(result.dataType);
        code = code.replace(`${type}(`, `${type}.fields(`);
      }
      code = code.replace(/\bcpuVector(?:_\d+)?\b/g, (name) => {
        const parts = this.#components(name);
        return parts ? `$vector([${parts.join(',')}])` : name;
      });
      return snip(code, result.dataType as d.BaseData, result.origin, result.possibleSideEffects);
    }

    if (
      result.origin !== 'constant' &&
      typeof result.dataType !== 'symbol' &&
      result.dataType.type !== 'ptr'
    ) {
      const code = this.ctx.resolveSnippet(result).value;
      const access = /^\(?([A-Za-z_]\w*)\)?(?:\.([xyzwrgba]{1,4})|\[(\d)\])$/.exec(code);
      const parts = access && this.#components(access[1]!);
      if (parts) {
        const selected = access![2]
          ? [...access![2]].map(
              (c) => parts['xyzw'.includes(c) ? 'xyzw'.indexOf(c) : 'rgba'.indexOf(c)]!,
            )
          : [parts[Number(access![3])]!];
        return snip(
          selected.length === 1 ? selected[0]! : this.#vector(result.dataType, selected),
          result.dataType,
          result.origin,
          result.possibleSideEffects,
        );
      }
    }

    if (['mutable', 'readonly', 'uniform'].includes(result.origin)) {
      const code = this.ctx.resolveSnippet(result).value;
      if (this.#addresses.has(code)) return result;
      const indexed = /^(\w+)\[(.*)\]$/.exec(code);
      const field = /^(\w+)\.([A-Za-z_]\w*)$/.exec(code);
      const base = this.#addresses.get(indexed?.[1] ?? field?.[1] ?? '');
      if (base) {
        const member = field && base.layout.fields?.[field[2]!];
        const component =
          field && /^[xyzw]$/.test(field[2]!) ? 'xyzw'.indexOf(field[2]!) : undefined;
        const index = indexed?.[2] ?? component;
        const layout = member ? member.layout : base.layout.element;
        if (layout && (member || index !== undefined)) {
          const offset = member
            ? `(${base.offset})+${member.offset}`
            : `(${base.offset})+(${index})*${base.layout.stride}`;
          const address = { layout, offset, base: base.base };
          if (['u32', 'i32', 'f32'].includes(layout.type))
            return snip(
              readAddress(address),
              result.dataType as d.BaseData,
              result.origin,
              result.possibleSideEffects,
            );
          const name = this.ctx.makeUniqueIdentifier('cpuAddress', 'global');
          this.#addresses.set(name, address);
          return snip(
            name,
            result.dataType as d.BaseData,
            result.origin,
            result.possibleSideEffects,
          );
        }
      }
    }
    if (
      Array.isArray(expression) &&
      expression[0] === NODE.unaryExpr &&
      scalar(result.dataType).startsWith('vec')
    ) {
      if (expression[1] !== '-') unsupported('vector unary operator');
      // eslint-disable-next-line no-underscore-dangle
      const argument = this._expression(expression[2]);
      const count = Number(scalar(result.dataType)[3]);
      return snip(
        `((v)=>$vector([${Array.from({ length: count }, (_, i) => `-v[${i}]`).join(',')}]))(${this.#value(argument)})`,
        result.dataType as d.BaseData,
        'runtime',
        result.possibleSideEffects,
      );
    }
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

  protected override _return(statement: Return): string {
    // eslint-disable-next-line no-underscore-dangle
    const code = super._return(statement);
    const type =
      this.ctx.topFunctionReturnType ??
      [...(this.ctx.topFunctionScope?.reportedReturnTypes ?? [])].at(-1);
    if (!type || type.type === 'void') return code;
    const match = /^(\s*return )([\s\S]*);$/.exec(code);
    if (!match) return code;
    const ctor = this.emitTypeAnnotation(type);
    const address = this.#addresses.get(match[2]!);
    const value = address ? readAddress(address) : match[2]!;
    return `${match[1]}${ctor.startsWith('cpuType') ? `${ctor}(${value})` : value};`;
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
    const code = this.ctx.resolveSnippet(value).value;
    const address = this.#addresses.get(code);
    if (address) {
      const name = this.ctx.makeUniqueIdentifier(rawId, 'block');
      this.#addresses.set(name, { ...address, offset: name });
      this.ctx.defineVariable(rawId, snip(name, value.dataType as d.BaseData, value.origin, false));
      return { code: `const ${name} = ${address.offset};`, definesInNearestScope: true };
    }
    const components = this.#parts(value);
    if (components) {
      const name = this.ctx.makeUniqueIdentifier(rawId, 'block');
      this.#vectors.set(name, components);
      this.ctx.defineVariable(rawId, snip(name, value.dataType as d.BaseData, value.origin, false));
      return { code: '', definesInNearestScope: false };
    }
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
      const target = lowerReferences(this.#value(args[0]!));
      const parts = addressParts(target);
      const object =
        parts?.object ??
        (/^\w+$/.test(target)
          ? `({get value(){return ${target}},set value(v){${target}=v}})`
          : undefined);
      const key = parts?.key ?? (/^\w+$/.test(target) ? '"value"' : undefined);
      if (!object || key === undefined) unsupported(`atomic reference ${target}`);
      let schema = args[0]!.dataType as d.BaseData;
      if (d.isPtr(schema)) schema = schema.inner;
      if (!d.isAtomic(schema)) unsupported(`atomic type ${schema.type}`);
      const inner = schema.inner;
      const values = args.slice(1).map((arg) => this.#value(arg));
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
        atomicCompareExchangeWeak:
          'const exchanged=old===v;if(exchanged)a[k]=w;return {old_value:old,exchanged};',
      };
      let operation = operations[name];
      if (operation)
        operation = operation.replace(
          /a\[k\]=([^;]+);/g,
          `a[k]=(($1) ${inner.type === 'u32' ? '>>>0' : '|0'});`,
        );
      if (!operation) unsupported(`builtin ${name}`);
      return snip(
        `((a,k,v,w)=>{const old=a[k];${operation}})(${object},${key},${values[0] ?? '0'},${values[1] ?? '0'})`,
        name === 'atomicStore'
          ? d.Void
          : name === 'atomicCompareExchangeWeak'
            ? d.struct({ old_value: inner, exchanged: d.bool })
            : inner,
        'runtime',
        true,
      );
    }
    if (name === 'workgroupUniformLoad') {
      let type = args[0]!.dataType as d.BaseData;
      if (d.isPtr(type)) type = type.inner;
      if (d.isAtomic(type)) type = type.inner;
      const code = lowerReferences(this.#value(args[0]!));
      return snip(
        `(yield* (function*(){yield;const value=${code};yield;return value;})())`,
        type,
        'runtime',
        true,
      );
    }
    const passed =
      Object.values(std).includes(callee as never) || 'type' in callee
        ? args
        : args.map((arg) => {
            if (typeof arg.dataType !== 'symbol' && arg.dataType.type !== 'ptr') {
              const code = this.ctx.resolveSnippet(arg).value;
              if (this.#addresses.has(code) || this.#components(code))
                return snip(this.#value(arg), arg.dataType, 'runtime', arg.possibleSideEffects);
            }
            return arg;
          });
    const result = super.call(callee, passed);
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
        'eq',
        'ne',
        'lt',
        'le',
        'gt',
        'ge',
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
    if (type.startsWith('texture') || type.startsWith('sampler')) return 'cpuHandle';
    if (['u32', 'i32', 'f32', 'bool', 'void', 'abstractInt', 'abstractFloat'].includes(type)) {
      return type;
    }
    const known = this.#types.get(schema);
    if (known) return known;
    const layout = layoutOf(schema);
    const name = this.ctx.makeUniqueIdentifier('cpuType', 'global');
    this.#types.set(schema, name);
    if (/^vec[234][fui]$/.test(type)) {
      const count = Number(type[3]);
      const convert = (value: string) =>
        type.endsWith('u') ? `((${value})>>>0)` : type.endsWith('i') ? `((${value})|0)` : value;
      this.ctx.addDeclaration(
        `function ${name}(...args) { const value = args.length === 1 && typeof args[0] !== 'number' ? (args[0] ?? []) : args; return $vector([${Array.from({ length: count }, (_, i) => convert(`(value[${i}] ?? (value.length===1 ? value[0] : 0))`)).join(',')}]); }`,
        name,
      );
      return name;
    }
    if (schema.type === 'struct') {
      const entries = Object.entries((schema as d.WgslStruct).propTypes);
      const copied = entries.map(([key, field]) => {
        const fieldType = this.emitTypeAnnotation(field);
        const input = `value?.[${JSON.stringify(key)}]`;
        const copy = fieldType.startsWith('cpuType')
          ? `${fieldType}(${input})`
          : fieldType === 'u32'
            ? `((${input})>>>0)`
            : fieldType === 'i32'
              ? `((${input})|0)`
              : fieldType === 'bool'
                ? `Boolean(${input})`
                : `Number(${input}??0)`;
        return `${JSON.stringify(key)}:${copy}`;
      });
      this.ctx.addDeclaration(
        `function ${name}(value) { return {${copied.join(',')}}; }\n${name}.fields=(...args)=>${name}({${entries.map(([key], i) => `${JSON.stringify(key)}:args[${i}]`).join(',')}});`,
        name,
      );
      return name;
    }
    const layoutName = this.ctx.makeUniqueIdentifier('cpuLayout', 'global');
    this.ctx.addDeclaration(
      `const ${layoutName}=${JSON.stringify(layout)};\nfunction ${name}(...args) { return $construct(${layoutName}, ...args); }${
        layout.fields
          ? `\n${name}.fields=(...args)=>${name}({${Object.keys(layout.fields)
              .map((key, i) => `${JSON.stringify(key)}:args[${i}]`)
              .join(',')}});`
          : ''
      }`,
      name,
    );
    return name;
  }

  override typeInstantiation(schema: d.BaseData, args: readonly Snippet[]): ResolvedSnippet {
    if (
      /^vec[234][fui]$/.test(schema.type) &&
      args.some((arg) => arg.origin !== 'constant') &&
      args.every(
        (arg) =>
          !arg.possibleSideEffects &&
          (scalar(arg.dataType).startsWith('vec') ? !!this.#parts(arg) : true),
      )
    ) {
      const parts = args.flatMap((arg) => this.#parts(arg) ?? [this.#value(arg)]);
      const count = Number(schema.type[3]);
      const components = Array.from({ length: count }, (_, i) => {
        const value = parts.length === 1 ? parts[0]! : (parts[i] ?? '0');
        return schema.type.endsWith('u')
          ? `((${value})>>>0)`
          : schema.type.endsWith('i')
            ? `((${value})|0)`
            : value;
      });
      return snip(this.#vector(schema, components), schema, 'runtime', false);
    }
    const vectorType = /^vec([234])([fui])$/.exec(schema.type);
    if (vectorType) {
      const count = Number(vectorType[1]);
      const components: string[] = [];
      args.forEach((arg, index) => {
        const width = /^vec([234])/.exec(scalar(arg.dataType));
        if (width) for (let i = 0; i < Number(width[1]); i++) components.push(`a${index}[${i}]`);
        else components.push(`a${index}`);
      });
      const values = Array.from({ length: count }, (_, i) => {
        const value = components.length === 1 ? components[0]! : (components[i] ?? '0');
        return vectorType[2] === 'u'
          ? `((${value})>>>0)`
          : vectorType[2] === 'i'
            ? `((${value})|0)`
            : value;
      });
      return snip(
        `((${args.map((_, i) => `a${i}`).join(',')})=>$vector([${values.join(',')}]))(${args.map((arg) => this.#value(arg)).join(',')})`,
        schema,
        'runtime',
        args.some((arg) => arg.possibleSideEffects),
      );
    }
    const type = this.emitTypeAnnotation(schema);
    const value = args.length ? this.#value(args[0]!) : '0';
    if (schema.type === 'struct' && args.length) {
      const keys = Object.keys((schema as d.WgslStruct).propTypes);
      return snip(
        `${type}({${args.map((arg, i) => `${JSON.stringify(keys[i])}:${this.#value(arg)}`).join(',')}})`,
        schema,
        'runtime',
        args.some((arg) => arg.possibleSideEffects),
      );
    }
    if (type.startsWith('cpuType'))
      return snip(
        `${type}(${args.map((arg) => this.#value(arg)).join(',')})`,
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
    let components = this.#components(rhs);
    let prefix = '';
    if (!components && /^vec[234][fui]$/.test(schema.type)) {
      const address = this.#addresses.get(rhs);
      if (address) components = this.#parts(snip(rhs, schema, 'runtime', false));
      if (!components) {
        const temporary = this.ctx.makeUniqueIdentifier('cpuValue', 'global');
        prefix = `const ${temporary}=${rhs};`;
        components = Array.from({ length: Number(schema.type[3]) }, (_, i) => `${temporary}[${i}]`);
      }
    }
    if (components) {
      const locals = components.map(() => this.ctx.makeUniqueIdentifier('cpuComponent', 'global'));
      this.#vectors.set(name, locals);
      return prefix + locals.map((local, i) => `let ${local}=${components[i]};`).join('');
    }
    const type = this.emitTypeAnnotation(schema);
    const address = this.#addresses.get(rhs);
    if (address) rhs = readAddress(address);
    return `let ${name} = ${type.startsWith('cpuType') ? `${type}(${rhs})` : rhs};`;
  }

  override declareGlobalConst(options: ConstantDefinitionOptions): ResolvedSnippet {
    this.emitTypeAnnotation(options.dataType);
    this.ctx.addDeclaration(`const ${options.id} = ${this.#value(options.init)};`, options.id);
    return snip(options.id, options.dataType, 'constant-immutable-def');
  }

  override declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    const { dataType, id, scope, group, binding } = options;
    if (scope === 'handle') {
      const kind = dataType.type.startsWith('sampler')
        ? 'sampler'
        : dataType.type.startsWith('texture_storage')
          ? 'storageTexture'
          : 'texture';
      if (
        kind !== 'sampler' &&
        !['2d', '2d-array', '3d', 'cube', 'cube-array'].includes(
          (dataType as d.WgslTexture).dimension,
        )
      )
        unsupported(`texture dimension ${(dataType as d.WgslTexture).dimension}`);
      const access = (dataType as d.WgslStorageTexture).access;
      this.ctx.addDeclaration(
        `//#binding ${JSON.stringify({ group, binding, kind, reads: kind !== 'sampler' && access !== 'write-only', writes: kind === 'storageTexture' && access !== 'read-only' })}\nconst ${id} = $resources.${kind === 'sampler' ? 'sampler' : 'texture'}(${group},${binding});`,
        id,
      );
      return snip(id, dataType, 'handle');
    }
    if (scope === 'private' || scope === 'workgroup') {
      const ctor = this.emitTypeAnnotation(dataType);
      const initial = options.init
        ? this.#value(options.init)
        : ctor.startsWith('cpuType')
          ? `${ctor}()`
          : '0';
      if (scope === 'private') this.#privateResets.push(`${id}=${initial};`);
      const scalarType = ['u32', 'i32', 'f32', 'bool'].includes(scalar(dataType));
      const expression =
        scope === 'private'
          ? initial
          : `$shared(${JSON.stringify(id)},()=>(${scalarType ? `{value:${initial}}` : initial}))`;
      this.ctx.addDeclaration(
        `//#scope ${scope}\n${scope === 'private' ? `//#private ${id}\n` : ''}let ${id} = ${expression};`,
        id,
      );
      return snip(scope === 'workgroup' && scalarType ? `${id}.value` : id, dataType, scope);
    }
    if (!['uniform', 'readonly', 'mutable'].includes(scope)) unsupported(`${scope} variable ${id}`);
    const array = dataType.type === 'array' ? (dataType as d.WgslArray) : undefined;
    const element = array?.elementType ?? dataType;
    const type = scalar(d.isAtomic(element) ? element.inner : element);
    if (!['u32', 'i32', 'f32'].includes(type)) {
      const layout = layoutOf(dataType);
      this.ctx.addDeclaration(
        `//#binding ${JSON.stringify({ group, binding, kind: 'buffer', reads: true, writes: scope === 'mutable' })}\nconst ${id}_u32 = $buffer(${group},${binding},Uint32Array);\nconst ${id}_i32 = $buffer(${group},${binding},Int32Array);\nconst ${id}_f32 = $buffer(${group},${binding},Float32Array);`,
        id,
      );
      const address = { layout, base: id, offset: '0' };
      this.#globalAddresses.set(id, address);
      this.#addresses.set(id, address);
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
    const previousVectors = this.#vectors;
    const previousAddresses = this.#addresses;
    this.#vectors = new Map();
    this.#addresses = new Map(this.#globalAddresses);
    try {
      if (options.functionType !== 'normal' && options.functionType !== 'compute')
        unsupported(`${options.functionType} entry point`);
      this.#functions.add(options.name);
      const argumentCopies = new Map<string, string>();
      if (options.functionType === 'normal')
        for (const arg of options.args) {
          const type = scalar(arg.decoratedType);
          if (/^vec[234][fui]$/.test(type)) {
            const parts = Array.from({ length: Number(type[3]) }, () =>
              this.ctx.makeUniqueIdentifier('cpuArgument', 'global'),
            );
            this.#vectors.set(arg.name, parts);
            argumentCopies.set(
              arg.name,
              parts.map((part, i) => `let ${part}=${arg.name}[${i}];`).join(''),
            );
          }
        }
      // The protected AST emitter is the extension hook TypeGPU exposes.
      // eslint-disable-next-line no-underscore-dangle
      let body = this._block(options.body, false).code;
      for (const [name, address] of this.#addresses) {
        if (address.layout.type === 'array') {
          const length =
            address.layout.count ||
            `Math.floor((${address.base}_u32.byteLength-(${address.offset}))/${address.layout.stride})`;
          body = body
            .replaceAll(`arrayLength((&${name}))`, String(length))
            .replaceAll(`arrayLength(&${name})`, String(length));
        }
        const local = { ...address, offset: 'offset' };
        if (body.includes(`(&${name})`))
          body = body.replaceAll(
            `(&${name})`,
            `((offset)=>({get $(){return ${readAddress(local)}},set $(value){${writeAddress(local, 'value')}}}))(${address.offset})`,
          );
        if (body.includes(`return ${name};`))
          body = body.replaceAll(`return ${name};`, `return ${readAddress(address)};`);
      }
      body = lowerReferences(body);
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
      const generator = /\(yield(?:\)|\* )/.test(body);
      if (generator) this.#generators.add(options.name);
      body = body.replace(/\bcpuVector(?:_\d+)?\b/g, (name) => {
        const parts = this.#components(name);
        return parts ? `$vector([${parts.join(',')}])` : name;
      });
      body = body.replace(/\bcpuAddress(?:_\d+)?\b/g, (name) => {
        const address = this.#addresses.get(name);
        return address ? readAddress(address) : name;
      });
      const returnType = options.determineReturnType();
      this.emitTypeAnnotation(returnType);
      const args = options.args.filter((arg) => arg.used || options.functionType === 'normal');
      if (options.functionType === 'normal') {
        const copies = args
          .map((arg) => {
            if (argumentCopies.has(arg.name)) return argumentCopies.get(arg.name)!;
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
      if (this.#privateResets.length) body = body.replace('{', `{${this.#privateResets.join('')}`);
      const size = [1, 1, 1];
      options.workgroupSize?.forEach((value, index) => {
        size[index] = value;
      });
      return `//#kit-gl compute\n//#entry ${JSON.stringify({ name: options.name, size, inputs, generator })}\nfunction${generator ? '*' : ''} ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}`;
    } finally {
      this.#vectors = previousVectors;
      this.#addresses = previousAddresses;
      for (const [id, address] of this.#globalAddresses) this.#addresses.set(id, address);
    }
  }

  override emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
    const type = scalar(lhs.dataType);
    const rhsType = scalar(rhs.dataType);
    const leftParts = this.#parts(lhs),
      rightParts = this.#parts(rhs);
    const vector = type.startsWith('vec') ? lhs : rhsType.startsWith('vec') ? rhs : undefined;
    const width = vector ? Number(scalar(vector.dataType)[3]) : 0;
    const targetAddress = this.#addresses.get(this.ctx.resolveSnippet(lhs).value);
    if (vector && leftParts && ['+=', '-=', '*=', '/=', '%='].includes(op)) {
      const value = this.emitBinaryOp(lhs, op.slice(0, -1) as BinaryOperator, rhs);
      return this.emitBinaryOp(
        lhs,
        '=',
        snip(value, lhs.dataType as d.BaseData, 'runtime', rhs.possibleSideEffects),
      );
    }
    if (vector && op === '=' && leftParts && !rightParts && !targetAddress)
      return `((v)=>{${leftParts.map((part, i) => `${part}=v[${i}];`).join('')}})(${this.#value(rhs)})`;
    if (vector && op === '=' && rightParts) {
      if (targetAddress) {
        const parameters = rightParts.map((_, i) => `v${i}`);
        const writes = parameters
          .map((value, i) =>
            writeAddress(
              {
                layout: targetAddress.layout.element!,
                base: targetAddress.base,
                offset: `o+${i * 4}`,
              },
              value,
            ),
          )
          .join('');
        return `((o,${parameters.join(',')})=>{${writes}})(${targetAddress.offset},${rightParts.join(',')})`;
      }
      if (leftParts)
        return `((${rightParts.map((_, i) => `v${i}`).join(',')})=>(${leftParts.map((part, i) => `${part}=v${i}`).join(',')}))(${rightParts.join(',')})`;
    }
    if (
      vector &&
      !type.startsWith('mat') &&
      !rhsType.startsWith('mat') &&
      (!type.startsWith('vec') || leftParts) &&
      (!rhsType.startsWith('vec') || rightParts) &&
      !lhs.possibleSideEffects &&
      !rhs.possibleSideEffects &&
      op !== '='
    ) {
      const primitive = scalar(vector.dataType).endsWith('u')
        ? d.u32
        : scalar(vector.dataType).endsWith('i')
          ? d.i32
          : d.f32;
      const assignment = op.endsWith('=') && !['==', '!=', '<=', '>='].includes(op);
      const operator = (assignment ? op.slice(0, -1) : op) as BinaryOperator;
      const components = Array.from({ length: width }, (_, i) =>
        this.emitBinaryOp(
          snip(leftParts?.[i] ?? this.#value(lhs), primitive, 'runtime', false),
          operator,
          snip(rightParts?.[i] ?? this.#value(rhs), primitive, 'runtime', false),
        ),
      );
      const result = snip(
        this.#vector(vector.dataType as d.BaseData, components),
        vector.dataType as d.BaseData,
        'runtime',
        false,
      );
      return assignment
        ? this.emitBinaryOp(lhs, '=', result)
        : this.ctx.resolveSnippet(result).value;
    }
    const target = this.#addresses.get(this.ctx.resolveSnippet(lhs).value);
    if (target && op === '=')
      return `((offset,value)=>{${writeAddress({ ...target, offset: 'offset' }, 'value')}return value;})(${target.offset},${this.#value(rhs)})`;
    if (target && ['+=', '-=', '*=', '/=', '%='].includes(op)) {
      const local = { ...target, offset: 'offset' };
      const value = this.emitBinaryOp(
        snip(readAddress(local), lhs.dataType as d.BaseData, 'runtime'),
        op.slice(0, -1) as BinaryOperator,
        snip('rhs', rhs.dataType as d.BaseData, 'runtime'),
      );
      return `((offset,rhs)=>{const value=${value};${writeAddress(local, 'value')}return value;})(${target.offset},${this.#value(rhs)})`;
    }

    if (type.startsWith('mat') || rhsType.startsWith('mat')) {
      const left = this.#value(lhs),
        right = this.#value(rhs);
      if (op === '=')
        return `${left}=${this.emitTypeAnnotation(lhs.dataType as d.BaseData)}(${right})`;
      const lm = /^mat([234])x([234])/.exec(type),
        rm = /^mat([234])x([234])/.exec(rhsType);
      let output: string;
      if (op === '*' && lm && rhsType.startsWith('vec'))
        output = `$vector([${Array.from({ length: Number(lm[2]) }, (_r, r) => Array.from({ length: Number(lm[1]) }, (_c, c) => `a[${c}][${r}]*b[${c}]`).join('+')).join(',')}])`;
      else if (op === '*' && rm && type.startsWith('vec'))
        output = `$vector([${Array.from({ length: Number(rm[1]) }, (_c, c) => Array.from({ length: Number(rm[2]) }, (_r, r) => `a[${r}]*b[${c}][${r}]`).join('+')).join(',')}])`;
      else if (op === '*' && lm && rm)
        output = `[${Array.from({ length: Number(rm[1]) }, (_c, c) => `$vector([${Array.from({ length: Number(lm[2]) }, (_r, r) => Array.from({ length: Number(lm[1]) }, (_k, k) => `a[${k}][${r}]*b[${c}][${k}]`).join('+')).join(',')}])`).join(',')}]`;
      else if (['+', '-', '*', '/'].includes(op)) {
        const shape = lm ?? rm!;
        output = `[${Array.from({ length: Number(shape[1]) }, (_c, c) => `$vector([${Array.from({ length: Number(shape[2]) }, (_r, r) => `${lm ? `a[${c}][${r}]` : 'a'}${op}${rm ? `b[${c}][${r}]` : 'b'}`).join(',')}])`).join(',')}]`;
      } else return unsupported(`matrix operator ${op}`);
      return `((a,b)=>${output})(${left},${right})`;
    }
    if (type.startsWith('vec') || rhsType.startsWith('vec')) {
      const schema = (type.startsWith('vec') ? lhs.dataType : rhs.dataType) as d.BaseData;
      const vectorType = scalar(schema);
      const size = Number(vectorType[3]);
      const primitive = vectorType.endsWith('u') ? d.u32 : vectorType.endsWith('i') ? d.i32 : d.f32;
      const left = this.#value(lhs);
      const right = this.#value(rhs);
      const ctor = this.emitTypeAnnotation(schema);
      if (op === '=') return `${left} = ${rhs.origin === 'runtime' ? right : `${ctor}(${right})`}`;
      const assignment = op.endsWith('=') && !['==', '!=', '<=', '>='].includes(op);
      const operation = (assignment ? op.slice(0, -1) : op) as BinaryOperator;
      const components = Array.from({ length: size }, (_, i) =>
        this.emitBinaryOp(
          snip(type.startsWith('vec') ? `a[${i}]` : 'a', primitive, 'runtime'),
          operation,
          snip(rhsType.startsWith('vec') ? `b[${i}]` : 'b', primitive, 'runtime'),
        ),
      );
      const code = `((a,b)=>$vector([${components.join(',')}]))(${left},${right})`;
      return assignment ? `${left} = ${code}` : code;
    }
    if (op === '=' && ['struct', 'array'].includes(type))
      return `${this.#value(lhs)} = ${this.emitTypeAnnotation(lhs.dataType as d.BaseData)}(${this.#value(rhs)})`;

    if (!['u32', 'i32', 'f32', 'bool', 'abstractInt', 'abstractFloat'].includes(type))
      unsupported(`${op} on ${type}`);
    const left = this.#value(lhs);
    const right = this.#value(rhs);
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
    const packing = /^(pack|unpack)(?:2x16|4x8)(float|snorm|unorm)$/.exec(name);
    if (packing)
      return `$${packing[1]}(${this.#value(args[0]!)},${name.includes('16') ? 16 : 8},${JSON.stringify(packing[2])})`;
    if (name === 'quantizeToF16')
      return `$unpack($pack([${this.#value(args[0]!)},0],16,'float'),16,'float')[0]`;
    if (name === 'transpose') {
      const type = scalar(args[0]!.dataType),
        columns = Number(type[3]),
        rows = Number(type[5]);
      return `((m)=>[${Array.from({ length: rows }, (_r, r) => `$vector([${Array.from({ length: columns }, (_c, c) => `m[${c}][${r}]`).join(',')}])`).join(',')}])(${this.#value(args[0]!)})`;
    }
    if (/^bitcast(F32|I32|U32)to(F32|I32|U32)$/.test(name)) {
      const match = /^bitcast(F32|I32|U32)to(F32|I32|U32)$/.exec(name)!;
      return `$bits.${match[1]!.toLowerCase()}_${match[2]!.toLowerCase()}(${this.#value(args[0]!)})`;
    }
    if (name.startsWith('texture')) {
      const supported = [
        'textureLoad',
        'textureStore',
        'textureDimensions',
        'textureNumLevels',
        'textureNumLayers',
        'textureSampleLevel',
        'textureSampleCompareLevel',
      ];
      if (!supported.includes(name)) unsupported(`builtin ${name}`);
      return `$textures.${name}(${args.map((arg) => this.#value(arg)).join(',')})`;
    }
    if (name === 'bitcast') {
      const target = this.ctx.resolveSnippet(templates[0]!).value;
      const source = scalar(args[0]!.dataType);
      const targetSchema = [...this.#types].find(([, typeName]) => typeName === target)?.[0];
      if (source.startsWith('vec') && targetSchema?.type.startsWith('vec')) {
        const from = source.endsWith('f') ? 'f32' : source.endsWith('i') ? 'i32' : 'u32';
        const to = targetSchema.type.endsWith('f')
          ? 'f32'
          : targetSchema.type.endsWith('i')
            ? 'i32'
            : 'u32';
        const count = Number(source[3]);
        if (count !== Number(targetSchema.type[3]))
          unsupported('bitcast with different vector sizes');
        return `((v)=>$vector([${Array.from({ length: count }, (_, i) => `$bits.${from}_${to}(v[${i}])`).join(',')}]))(${this.#value(args[0]!)})`;
      }
      if (!['u32', 'i32', 'f32'].includes(target) || !['u32', 'i32', 'f32'].includes(source))
        unsupported(`bitcast ${source} to ${target}`);
      return `$bits.${source}_${target}(${this.#value(args[0]!)})`;
    }
    if (templates.length) unsupported(`template call ${name}`);
    const values = args.map((arg) => this.#value(arg));

    if (name === 'all' || name === 'any') {
      const parts = this.#parts(args[0]!);
      if (parts) return `(${parts.join(name === 'all' ? '&&' : '||')})`;
    }
    if (name === 'all' || name === 'any')
      return scalar(args[0]!.dataType) === 'bool'
        ? values[0]!
        : `${values[0]}.${name === 'all' ? 'every' : 'some'}(Boolean)`;
    if (name === 'allEq') return `(${this.emitBinaryOp(args[0]!, '==', args[1]!)}).every(Boolean)`;
    const vectorArg = args.find((arg) => scalar(arg.dataType).startsWith('vec'));
    if (vectorArg) {
      const schema = vectorArg.dataType as d.BaseData;
      const count = Number(schema.type[3]);
      const ctor = this.emitTypeAnnotation(schema);
      const direct = args.map((arg) => this.#parts(arg));
      if (
        args.every(
          (arg, i) =>
            !arg.possibleSideEffects && (!scalar(arg.dataType).startsWith('vec') || direct[i]),
        )
      ) {
        const at = (arg: number, component: number) =>
          `(${direct[arg]?.[component] ?? this.#value(args[arg]!)})`;
        const product = Array.from(
          { length: count },
          (_, i) => `${at(0, i)}*${at(name === 'dot' || name === 'reflect' ? 1 : 0, i)}`,
        ).join('+');
        if (name === 'dot') return `(${product})`;
        if (name === 'length') return `Math.sqrt(${product})`;
        if (name === 'distance')
          return `Math.sqrt(${Array.from({ length: count }, (_, i) => `(${at(0, i)}-${at(1, i)})**2`).join('+')})`;
        let parts: string[];
        if (name === 'normalize')
          parts = Array.from({ length: count }, (_, i) => `${at(0, i)}/Math.sqrt(${product})`);
        else if (name === 'reflect')
          parts = Array.from({ length: count }, (_, i) => `${at(0, i)}-2*(${product})*${at(1, i)}`);
        else if (name === 'cross')
          parts = [0, 1, 2].map(
            (i) =>
              `${at(0, (i + 1) % 3)}*${at(1, (i + 2) % 3)}-${at(0, (i + 2) % 3)}*${at(1, (i + 1) % 3)}`,
          );
        else
          parts = Array.from({ length: count }, (_, i) =>
            this.emitCall(
              name,
              [],
              args.map((arg, j) =>
                snip(
                  at(j, i),
                  scalar(arg.dataType).startsWith('vec') ? d.f32 : (arg.dataType as d.BaseData),
                  'runtime',
                  false,
                ),
              ),
            ),
          );
        return this.#vector(schema, parts);
      }
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
      if (name === 'reflect') {
        const product = Array.from({ length: count }, (_, i) => `a0[${i}]*a1[${i}]`).join('+');
        body = `$vector([${Array.from({ length: count }, (_, i) => `a0[${i}]-2*(${product})*a1[${i}]`).join(',')}])`;
      } else if (name === 'distance')
        body = `Math.sqrt(${Array.from({ length: count }, (_, i) => `(a0[${i}]-a1[${i}])**2`).join('+')})`;
      else if (name === 'dot' || name === 'length')
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
    if (name === 'reverseBits')
      return `((a)=>{let r=0;for(let i=0;i<32;i++){r=(r<<1)|(a&1);a>>>=1;}return r>>>0;})(${values[0]})`;
    if (name === 'countOneBits')
      return `((a)=>{a=a-((a>>>1)&0x55555555);a=(a&0x33333333)+((a>>>2)&0x33333333);return Math.imul((a+(a>>>4))&0x0f0f0f0f,0x01010101)>>>24;})(${values[0]})`;
    const expressions: Record<string, string> = {
      not: '!a',
      smoothstep: '((t)=>t*t*(3-2*t))(Math.min(Math.max((c-a)/(b-a),0),1))',
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
        'cosh',
        'sinh',
        'tanh',
        'acosh',
        'asinh',
        'atanh',
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
      if (name === 'round')
        return `((a)=>{const f=Math.floor(a);return a-f===0.5?(f%2===0?f:f+1):Math.round(a);})(${values[0]})`;
      return `Math.${name}(${values.join(',')})`;
    }
    if (name === 'select') return `(${values[2]} ? ${values[1]} : ${values[0]})`;
    return unsupported(`builtin ${name}`);
  }
}
