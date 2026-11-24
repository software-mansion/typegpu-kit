import { expandBarrierHelpers, type BarrierHelper } from './phase-helpers.ts';
import { splitPhases } from './phases.ts';
import { readAddress, referenceAddress, writeAddress, type Address } from './address.ts';
import { addressParts, callArguments, lowerReferences } from './references.ts';
import { layoutOf } from './memory.ts';
import { NodeTypeCatalog as NODE, type Expression, type Statement, type Return } from 'tinyest';
import { d, std } from 'typegpu';
import {
  WgslGenerator,
  makeResolvable,
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
  #functionEffects = new Map<string, boolean>();
  #vectorReturns = new Map<
    string,
    { schema: d.BaseData; parameters: string[]; components: string[] }
  >();
  #functionUsed = new Map<string, boolean[]>();
  #effects = false;
  #functionVectors = new Map<string, (number | undefined)[]>();
  #functionArguments = new Map<string, d.BaseData[]>();
  #privateResets: string[] = [];
  #addresses = new Map<string, Address>();
  #globalAddresses = new Map<string, Address>();
  #vectors = new Map<string, string[]>();
  #constantVectors = new Map<string, string[]>();
  #ownedLocals = new Set<string>();
  #lowerSwizzle(original: Snippet, code: string): Snippet {
    if (typeof original.value !== 'object' || !original.value)
      return snip(code, original.dataType, original.origin, original.possibleSideEffects);
    // Keep the Swizzle identity that the base generator's mutation validation inspects.
    const value = Object.assign(
      Object.create(Object.getPrototypeOf(original.value)) as object,
      original.value,
    );
    makeResolvable(value, {
      asString: () => 'CPU swizzle',
      resolve: () => ({ ...original, value: code, dataType: original.dataType as d.BaseData }),
    });
    return snip(value, original.dataType, original.origin, original.possibleSideEffects);
  }
  #components(code: string): string[] | undefined {
    const name = code.replace(/^\((\w+)\)$/, '$1');
    return this.#vectors.get(name) ?? this.#constantVectors.get(name);
  }
  #vector(_schema: d.BaseData, components: string[]): string {
    const name = this.ctx.makeUniqueIdentifier('cpuVector', 'global');
    this.#vectors.set(name, components);
    return name;
  }
  #parts(snippet: Snippet): string[] | undefined {
    if (typeof snippet.dataType === 'symbol' || !scalar(snippet.dataType).startsWith('vec'))
      return undefined;
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
    if (/^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*|\[\d+\])*$/.test(code))
      return Array.from({ length: Number(scalar(snippet.dataType)[3]) }, (_, i) => `${code}[${i}]`);
    if (
      !snippet.possibleSideEffects &&
      /^[\w$.[\]()+\-*/%<>&|^ ]+$/.test(code) &&
      !/[\w$]\(/.test(code)
    )
      return Array.from(
        { length: Number(scalar(snippet.dataType)[3]) },
        (_, i) => `(${code})[${i}]`,
      );
    return undefined;
  }

  #matrixParts(snippet: Snippet): string[][] | undefined {
    if (snippet.possibleSideEffects) return undefined;
    const type = scalar(snippet.dataType);
    const shape = /^mat([234])x([234])f$/.exec(type);
    if (!shape) return undefined;
    const code = this.ctx.resolveSnippet(snippet).value;
    const address = this.#addresses.get(code);
    const columns = Number(shape[1]);
    const rows = Number(shape[2]);
    if (address?.layout.type === type)
      return Array.from({ length: columns }, (_, c) =>
        Array.from({ length: rows }, (_row, r) =>
          readAddress({
            layout: address.layout.element!.element!,
            base: address.base,
            offset: `(${address.offset})+${c * address.layout.stride! + r * 4}`,
          }),
        ),
      );
    if (/^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*|\[\d+\])*$/.test(code))
      return Array.from({ length: columns }, (_, c) =>
        Array.from({ length: rows }, (_row, r) => `${code}[${c}][${r}]`),
      );
    return undefined;
  }

  #value(snippet: Snippet): string {
    const code = this.ctx.resolveSnippet(snippet).value;
    const type = typeof snippet.dataType === 'symbol' ? '' : scalar(snippet.dataType);
    const components = type.startsWith('vec') ? this.#components(code) : undefined;
    if (components) return `$vector([${components.join(',')}])`;
    const address = this.#addresses.get(code);
    return address && address.layout.type === type ? readAddress(address) : code;
  }
  #fresh(code: string): boolean {
    if (code.startsWith('({')) return true;
    if (
      /^(?:cpuVector(?:_\d+)?$|cpuType(?:_\d+)?(?:\.fields)?\(|\$vector\(|\(\([^)]*\)=>)/.test(code)
    )
      return true;
    const call = /^(\w+)\(/.exec(code);
    return !!call && this.#functions.has(call[1]!);
  }
  #generators = new Set<string>();
  #barrierHelpers = new Map<string, BarrierHelper>();
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
      const components = scalar(result.dataType).startsWith('vec')
        ? this.#components(code)
        : undefined;
      if (components)
        return snip(
          this.#vector(result.dataType, components),
          result.dataType,
          result.origin,
          result.possibleSideEffects,
        );
      const address = this.#addresses.get(code);
      if (
        address &&
        address.layout.type === scalar(result.dataType) &&
        !/^cpuAddress(?:_\d+)?$/.test(code)
      ) {
        const name = this.ctx.makeUniqueIdentifier('cpuAddress', 'global');
        this.#addresses.set(name, address);
        return snip(name, result.dataType, result.origin, result.possibleSideEffects);
      }
    }
    if (
      Array.isArray(expression) &&
      (expression[0] === NODE.memberAccess || expression[0] === NODE.indexAccess) &&
      typeof result.dataType !== 'symbol' &&
      result.origin !== 'constant'
    ) {
      const code = this.ctx.resolveSnippet(result).value;
      const swizzle = /\.([xyzwrgba]{1,4})$/.exec(code);
      if (swizzle) {
        // eslint-disable-next-line no-underscore-dangle
        const target = this._expression(expression[1]);
        const parts = this.#parts(target);
        if (parts) {
          const selected = [...swizzle[1]!].map(
            (c) => parts['xyzw'.includes(c) ? 'xyzw'.indexOf(c) : 'rgba'.indexOf(c)]!,
          );
          return this.#lowerSwizzle(
            result,
            selected.length === 1
              ? selected[0]!
              : this.#vector(result.dataType as d.BaseData, selected),
          );
        }
        if (typeof target.dataType !== 'symbol' && scalar(target.dataType).startsWith('vec')) {
          const indices = [...swizzle[1]!].map((c) =>
            'xyzw'.includes(c) ? 'xyzw'.indexOf(c) : 'rgba'.indexOf(c),
          );
          const value = this.#value(target);
          return this.#lowerSwizzle(
            result,
            indices.length === 1
              ? `(${value})[${indices[0]}]`
              : `((v)=>$vector([${indices.map((i) => `v[${i}]`).join(',')}]))(${value})`,
          );
        }
      }
    }
    if (Array.isArray(expression) && expression[0] === NODE.objectExpr) {
      let code = this.ctx.resolveSnippet(result).value;
      if (typeof result.dataType !== 'symbol' && result.dataType.type === 'struct') {
        const type = this.emitTypeAnnotation(result.dataType);
        if (code.startsWith(`${type}(`)) {
          const values = callArguments(code);
          const fields = Object.entries((result.dataType as d.WgslStruct).propTypes);
          code = `({${fields
            .map(([key, schema], i) => {
              const value = values[i]!;
              const ctor = this.emitTypeAnnotation(schema);
              return `${JSON.stringify(key)}:${ctor.startsWith('cpuType') && !this.#fresh(value) ? `${ctor}(${value})` : value}`;
            })
            .join(',')}})`;
        }
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
    return `${match[1]}${ctor.startsWith('cpuType') && !address && !this.#ownedLocals.has(value) && !this.#fresh(value) ? `${ctor}(${value})` : value};`;
  }

  protected override _statement(statement: Statement): ResolvedStatement {
    if (Array.isArray(statement) && statement[0] === NODE.postUpdate) {
      // Preserve the base generator's mutability checks and variable tracking.
      // eslint-disable-next-line no-underscore-dangle
      super._statement(statement);
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
      this.#effects = true;
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
      const code = /^\w+$/.test(target)
        ? name === 'atomicLoad'
          ? target
          : `((v,w)=>{const old=${target};${operation.replaceAll('a[k]', target)}})(${values[0] ?? '0'},${values[1] ?? '0'})`
        : `((a,k,v,w)=>{const old=a[k];${operation}})(${object},${key},${values[0] ?? '0'},${values[1] ?? '0'})`;
      return snip(
        code,
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
      this.#effects = true;
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
              if (this.#addresses.has(code))
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
    if (['workgroupBarrier', 'storageBarrier', 'textureBarrier'].includes(name)) {
      this.#effects = true;
      return result;
    }
    if (Object.values(std).includes(callee as never)) {
      return snip(
        this.emitCall(name, [], args),
        result.dataType as d.BaseData,
        result.origin,
        result.possibleSideEffects,
      );
    }
    if (typeof result.dataType === 'symbol' || result.dataType.type === 'ptr') return result;
    const code = this.ctx.resolveSnippet(result).value;
    const resolvedName = /^(\w+)\(/.exec(code)?.[1];
    const argumentTypes = resolvedName && this.#functionArguments.get(resolvedName);
    if (resolvedName && argumentTypes) {
      const converted = callArguments(code).map((value, i) =>
        snip(
          value,
          argumentTypes[i]!,
          args[i]?.origin ?? 'runtime',
          args[i]?.possibleSideEffects ?? true,
        ),
      );
      return snip(
        this.emitCall(resolvedName, [], converted),
        result.dataType,
        result.origin,
        (this.#functionEffects.get(resolvedName) ?? true) ||
          args.some((arg) => arg.possibleSideEffects),
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
      const components = Array.from({ length: count }, (_, i) => `v${i}`);
      this.ctx.addDeclaration(
        `function ${name}(${components.map((component, i) => `${component}=${i === 0 ? '0' : 'v0'}`).join(',')}) { return typeof v0==='number' ? $vector([${components.map(convert).join(',')}]) : $vector([${components.map((_component, i) => convert(`(v0[${i}]??0)`)).join(',')}]); }`,
        name,
      );
      return name;
    }
    const matrix = /^mat([234])x([234])f$/.exec(type);
    if (matrix) {
      const columns = Number(matrix[1]),
        rows = Number(matrix[2]);
      const flat = Array.from(
        { length: columns },
        (_, c) =>
          `$vector([${Array.from({ length: rows }, (_row, r) => `(args[${c * rows + r}]??0)`).join(',')}])`,
      ).join(',');
      const copied = Array.from(
        { length: columns },
        (_, c) =>
          `$vector([${Array.from({ length: rows }, (_row, r) => `(columns[${c}]?.[${r}]??0)`).join(',')}])`,
      ).join(',');
      this.ctx.addDeclaration(
        `function ${name}(...args) { if(typeof args[0]==='number') return [${flat}]; const columns=args.length===1 ? (args[0]??[]) : args; return [${copied}]; }`,
        name,
      );
      return name;
    }
    if (d.isWgslArray(schema)) {
      const element = d.isAtomic(schema.elementType)
        ? schema.elementType.inner
        : schema.elementType;
      const ctor = this.emitTypeAnnotation(element);
      const input = 'value?.[i]';
      const copied = ctor.startsWith('cpuType')
        ? `${ctor}(${input})`
        : ctor === 'u32'
          ? `((${input})>>>0)`
          : ctor === 'i32'
            ? `((${input})|0)`
            : ctor === 'bool'
              ? `Boolean(${input})`
              : `Number(${input}??0)`;
      this.ctx.addDeclaration(
        `function ${name}(...args) { const value=args.length===1 ? args[0] : args; const result=new Array(${schema.elementCount}); for(let i=0;i<${schema.elementCount};i++) result[i]=${copied}; return result; }`,
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
      const name = this.#vector(schema, components);
      if (args.every((arg) => arg.origin === 'constant'))
        this.#constantVectors.set(name, components);
      return snip(name, schema, 'runtime', false);
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
      if (args.length === 1 && scalar(args[0]!.dataType) === 'struct')
        return snip(
          this.#fresh(value) || this.#addresses.has(this.ctx.resolveSnippet(args[0]!).value)
            ? value
            : `${type}(${value})`,
          schema,
          'runtime',
          args[0]!.possibleSideEffects,
        );
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
    this.#ownedLocals.add(name);
    return `let ${name} = ${type.startsWith('cpuType') && !address && !this.#fresh(rhs) ? `${type}(${rhs})` : rhs};`;
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
        `//#scope ${scope}\n${scope === 'private' ? `//#private ${id}\n` : `//#workgroup ${JSON.stringify({ name: id, initial: expression })}\n`}let ${id} = ${expression};`,
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
    const previousEffects = this.#effects;
    this.#effects = false;
    const previousVectors = this.#vectors;
    const previousAddresses = this.#addresses;
    const previousOwnedLocals = this.#ownedLocals;
    this.#ownedLocals = new Set();
    this.#vectors = new Map();
    this.#addresses = new Map(this.#globalAddresses);
    try {
      if (options.functionType !== 'normal' && options.functionType !== 'compute')
        unsupported(`${options.functionType} entry point`);
      this.#functions.add(options.name);
      const parameters = new Map<string, string[]>();
      this.#functionArguments.set(
        options.name,
        options.args.map((arg) => arg.decoratedType),
      );
      if (options.functionType === 'normal')
        for (const arg of options.args) {
          const type = scalar(arg.decoratedType);
          if (/^vec[234][fui]$/.test(type)) {
            const parts = Array.from({ length: Number(type[3]) }, () =>
              this.ctx.makeUniqueIdentifier('cpuArgument', 'global'),
            );
            this.#vectors.set(arg.name, parts);
            parameters.set(arg.name, parts);
          }
        }
      if (options.functionType === 'normal')
        this.#functionVectors.set(
          options.name,
          options.args.map((arg) => parameters.get(arg.name)?.length),
        );
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
        if (body.includes(`(&${name})`))
          body = body.replaceAll(`(&${name})`, referenceAddress(address));
        if (body.includes(`return ${name};`))
          body = body.replaceAll(`return ${name};`, `return ${readAddress(address)};`);
      }
      for (const [name, parts] of this.#vectors) {
        if (body.includes(`(&${name})`)) {
          const writes = parts.map((part, i) => `${part}=value[${i}];`).join('');
          const properties = parts
            .map((part, i) => `get ${i}(){return ${part}},set ${i}(value){${part}=value}`)
            .join(',');
          body = body.replaceAll(
            `(&${name})`,
            `((view)=>({get $(){return view},set $(value){${writes}}}))({length:${parts.length},${properties}})`,
          );
        }
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
      this.#functionEffects.set(options.name, this.#effects);
      this.#functionUsed.set(
        options.name,
        options.args.map((arg) => arg.used),
      );
      const returnType = options.determineReturnType();
      if (
        generator &&
        options.functionType === 'normal' &&
        ['u32', 'i32', 'f32', 'bool'].includes(scalar(returnType))
      )
        this.#barrierHelpers.set(options.name, {
          body,
          parameters: options.args.map((arg) => arg.name),
        });
      this.emitTypeAnnotation(returnType);
      const args = options.args.filter((arg) => arg.used || options.functionType === 'normal');
      const vectorReturn = /^\{\s*return \$vector\(\[([\s\S]*)\]\);\s*}$/.exec(body);
      if (!this.#effects && /^vec[234][fui]$/.test(scalar(returnType)) && vectorReturn) {
        const components = callArguments(`value(${vectorReturn[1]})`);
        if (components.length === Number(scalar(returnType)[3]))
          this.#vectorReturns.set(options.name, {
            schema: returnType,
            parameters: args.flatMap((arg) => parameters.get(arg.name) ?? [arg.name]),
            components,
          });
      }
      if (options.functionType === 'normal') {
        return `function${generator ? '*' : ''} ${options.name}(${args.flatMap((arg) => parameters.get(arg.name) ?? [arg.name]).join(',')}) ${body || '{}'}`;
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
      const phaseBody = expandBarrierHelpers(body, this.#barrierHelpers, () =>
        this.ctx.makeUniqueIdentifier('cpuPhaseLocal', 'global'),
      );
      const phases = generator
        ? splitPhases(
            phaseBody,
            args.map((arg) => arg.name),
          )
        : undefined;
      return `//#kit-gl compute\n//#entry ${JSON.stringify({ name: options.name, size, inputs, generator, ...(phases ? { phases: phases.count } : {}) })}\nfunction${generator ? '*' : ''} ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}${phases ? `\nconst ${options.name}_phases=${phases.source};` : ''}`;
    } finally {
      this.#effects = previousEffects;
      this.#vectors = previousVectors;
      this.#addresses = previousAddresses;
      this.#ownedLocals = previousOwnedLocals;
      for (const [id, address] of this.#globalAddresses) this.#addresses.set(id, address);
    }
  }

  override emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
    if (
      (op === '=' || (op.endsWith('=') && !['==', '!=', '<=', '>='].includes(op))) &&
      lhs.origin !== 'local-def'
    )
      this.#effects = true;
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
        return `${left}=${this.#fresh(right) ? right : `${this.emitTypeAnnotation(lhs.dataType as d.BaseData)}(${right})`}`;
      const lm = /^mat([234])x([234])/.exec(type),
        rm = /^mat([234])x([234])/.exec(rhsType);
      const matrixLeft = this.#matrixParts(lhs);
      const matrixRight = this.#matrixParts(rhs);
      if (op === '*' && matrixLeft && rightParts && !rhs.possibleSideEffects) {
        const rows = Number(lm![2]);
        return this.#vector(
          d[`vec${rows}f` as 'vec2f'],
          Array.from({ length: rows }, (_row, r) =>
            matrixLeft.map((column, c) => `(${column[r]})*(${rightParts[c]})`).join('+'),
          ),
        );
      }
      if (op === '*' && matrixRight && leftParts && !lhs.possibleSideEffects)
        return this.#vector(
          d[`vec${matrixRight.length}f` as 'vec2f'],
          matrixRight.map((column) =>
            column.map((value, r) => `(${leftParts[r]})*(${value})`).join('+'),
          ),
        );
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
    if (op === '=' && ['struct', 'array'].includes(type)) {
      const value = this.#value(rhs);
      return `${this.#value(lhs)} = ${this.#fresh(value) ? value : `${this.emitTypeAnnotation(lhs.dataType as d.BaseData)}(${value})`}`;
    }

    if (!['u32', 'i32', 'f32', 'bool', 'abstractInt', 'abstractFloat'].includes(type))
      unsupported(`${op} on ${type}`);
    const left = `(${this.#value(lhs)})`;
    const right = `(${this.#value(rhs)})`;
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
    if (name === 'textureStore') this.#effects = true;
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
        const components = this.#parts(args[0]!);
        if (components && !args[0]!.possibleSideEffects)
          return this.#vector(
            targetSchema,
            components.map((component) => `$bits.${from}_${to}(${component})`),
          );
        return `((v)=>$vector([${Array.from({ length: count }, (_, i) => `$bits.${from}_${to}(v[${i}])`).join(',')}]))(${this.#value(args[0]!)})`;
      }
      if (!['u32', 'i32', 'f32'].includes(target) || !['u32', 'i32', 'f32'].includes(source))
        unsupported(`bitcast ${source} to ${target}`);
      return `$bits.${source}_${target}(${this.#value(args[0]!)})`;
    }
    if (templates.length) unsupported(`template call ${name}`);
    const values = args.map((arg) => this.#value(arg));
    if (this.#functions.has(name)) {
      if (this.#functionEffects.get(name) ?? true) this.#effects = true;
      const used = this.#functionUsed.get(name);
      const ignored = args.map((arg, i) => used?.[i] === false && !arg.possibleSideEffects);
      const widths = this.#functionVectors.get(name) ?? [];
      const parts = args.map((arg) => this.#parts(arg));
      const inline = this.#vectorReturns.get(name);
      if (inline && args.every((arg, i) => !arg.possibleSideEffects && (!widths[i] || parts[i]))) {
        const expanded = args.flatMap((_arg, i) => (widths[i] ? parts[i]! : [values[i]!]));
        if (
          expanded.every(
            (value) => /^[\w$.[\]()+\-*/%<>&|^ ]+$/.test(value) && !/[\w$]\(/.test(value),
          )
        ) {
          const replacements = new Map(
            inline.parameters.map((parameter, i) => [parameter, `(${expanded[i]})`]),
          );
          return this.#vector(
            inline.schema,
            inline.components.map((component) =>
              component.replace(
                /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_$][\w$]*/g,
                (token, at: number) =>
                  component[at - 1] === '.' ? token : (replacements.get(token) ?? token),
              ),
            ),
          );
        }
      }
      const copied = args.map((arg, i) => {
        if (ignored[i]) return 'undefined';
        const type = scalar(arg.dataType);
        const value = values[i]!;
        if (!widths[i] && (['struct', 'array'].includes(type) || type.startsWith('mat'))) {
          // Non-pointer arguments are immutable; a callee cannot reach an owned caller local.
          if (
            !args.some((passed) => scalar(passed.dataType) === 'ptr') &&
            args.slice(i + 1).every((passed) => !passed.possibleSideEffects) &&
            (arg.origin === 'argument' || this.#ownedLocals.has(value))
          )
            return value;
          // Buffer reads and expression results already own their composite storage.
          if (this.#fresh(value) || value.startsWith('({') || value.startsWith('[')) return value;
          return `${this.emitTypeAnnotation(arg.dataType as d.BaseData)}(${value})`;
        }
        return value;
      });
      if (args.every((arg, i) => !widths[i] || (parts[i] && !arg.possibleSideEffects)))
        return `${name}(${args.flatMap((_, i) => (widths[i] ? (ignored[i] ? Array<string>(widths[i]!).fill('0') : parts[i]!) : [copied[i]!])).join(',')})`;
      // Evaluate all arguments in their original order before expanding vector components.
      const temporaries = args.map(() => this.ctx.makeUniqueIdentifier('cpuPassed', 'global'));
      const expanded = temporaries.flatMap((temporary, i) =>
        widths[i]
          ? Array.from({ length: widths[i]! }, (_, c) => `${temporary}[${c}]`)
          : [temporary],
      );
      const call = `${name}(${expanded.join(',')})`;
      return this.#generators.has(name)
        ? `(yield* (function*(${temporaries.join(',')}){return ${call};})(${copied.join(',')}))`
        : `((${temporaries.join(',')})=>${call})(${copied.join(',')})`;
    }

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
          (_, i) => `${at(0, i)}*${at(['dot', 'reflect', 'refract'].includes(name) ? 1 : 0, i)}`,
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
        else if (name === 'refract') {
          const eta = this.#value(args[2]!);
          parts = Array.from(
            { length: count },
            (_, i) =>
              `((k)=>k<0?0:((${eta})*${at(0, i)}-((${eta})*(${product})+Math.sqrt(k))*${at(1, i)}))(1-(${eta})*(${eta})*(1-(${product})*(${product})))`,
          );
        } else if (name === 'cross')
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
      } else if (name === 'refract') {
        const product = Array.from({ length: count }, (_, i) => `a0[${i}]*a1[${i}]`).join('+');
        body = `((dot)=>{const k=1-a2*a2*(1-dot*dot);return $vector([${Array.from({ length: count }, (_, i) => `k<0?0:a2*a0[${i}]-(a2*dot+Math.sqrt(k))*a1[${i}]`).join(',')}]);})(${product})`;
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
