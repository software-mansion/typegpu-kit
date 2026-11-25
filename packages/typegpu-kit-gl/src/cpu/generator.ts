import { f32, f32Literal, f32dot, f32op } from './floats.ts';
import { expandBarrierHelpers, type BarrierHelper } from './phase-helpers.ts';
import { splitPhases } from './phases.ts';
import { isPureExpression, prunePureLocals } from './dead-locals.ts';
import { completeCall, projectRecordBody, usedRecordFields } from './record-fields.ts';
import { readAddress, referenceAddress, writeAddress, type Address } from './address.ts';
import { addressParts, callArguments, lowerReferences } from './references.ts';
import { layoutOf } from './memory.ts';
import {
  NodeTypeCatalog as NODE,
  type Expression,
  type Statement,
  type Return,
  type Block,
  type Switch,
} from 'tinyest';
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

function integerCast(value: string, source: string, unsigned: boolean): string {
  const floating = /^(f32|f16|abstractFloat|vec[234][fh])$/.test(source);
  const converted = floating
    ? `Math.min(Math.max(${value},${unsigned ? 0 : -2147483648}),${unsigned ? 4294967040 : 2147483520})`
    : value;
  return `((${converted})${unsigned ? '>>>0' : '|0'})`;
}

const EXACT_FLOAT_BUILTINS = new Set([
  'abs',
  'ceil',
  'floor',
  'trunc',
  'round',
  'min',
  'max',
  'sign',
  'clamp',
  'saturate',
  'step',
  'select',
]);

export class CpuGenerator extends WgslGenerator {
  override languageKey = 'javascript';
  #functions = new Set<string>();
  #functionEffects = new Map<string, boolean>();
  #functionBodies = new Map<string, string>();
  #globalScopes = new Map<string, { scope: string; type: string }>();
  #uniformCaches = new Map<string, string>();
  #uniformValues = new Set<string>();
  #recordFunctions = new Map<string, { schema: d.WgslStruct; parameters: string[] }>();
  #recordProjections = new Map<string, string>();
  #recordArguments = new Map<string, (string[] | undefined)[]>();
  #parameterUse = new Map<string, boolean[]>();
  #projectArguments(name: string, values: string[]): string[] {
    const used = this.#parameterUse.get(name);
    const fields = this.#recordArguments.get(name);
    return values.map((value, i) => {
      if (used?.[i] === false && isPureExpression(value, this.#functionEffects)) return 'undefined';
      const wanted = fields?.[i];
      return wanted?.length ? (this.#projectRecord(value, wanted, false) ?? value) : value;
    });
  }
  #projectLocalRecords(body: string): string {
    const declarations = [...body.matchAll(/^(let|const) ([A-Za-z_$][\w$]*) = ([^\n]*);$/gm)];
    for (const local of declarations) {
      const fields = usedRecordFields(body.replace(local[0], ''), local[2]!);
      if (!fields?.length) continue;
      const projected = this.#projectRecord(local[3]!, fields, false);
      if (!projected) continue;
      if (
        !projected.startsWith('({') ||
        declarations.filter((other) => other[2] === local[2]).length !== 1
      ) {
        body = body.replace(local[0], `${local[1]} ${local[2]} = ${projected};`);
        continue;
      }
      const entries = callArguments(`record(${projected.slice(2, -2)})`).map((entry) =>
        /^"([A-Za-z_$][\w$]*)":([\s\S]*)$/.exec(entry),
      );
      if (entries.some((entry) => !entry)) continue;
      const replacements = new Map(
        entries.map((entry) => [entry![1]!, this.ctx.makeUniqueIdentifier('cpuField', 'global')]),
      );
      body = body.replace(
        local[0],
        entries.map((entry) => `let ${replacements.get(entry![1]!)} = ${entry![2]};`).join('\n'),
      );
      const name = local[2]!.replace(/[$]/g, '\\$');
      body = body.replace(
        new RegExp(`(?<![\\w$.])${name}\\.([A-Za-z_$][\\w$]*)`, 'g'),
        (access, field: string) => replacements.get(field) ?? access,
      );
    }
    return body;
  }

  #projectRecord(value: string, fields: readonly string[], unwrap: boolean): string | undefined {
    if (value.startsWith('({') && isPureExpression(value, this.#functionEffects)) {
      const projected = projectRecordBody(
        `{return ${value};}`,
        fields,
        unwrap,
        () => '0',
        () => undefined,
      );
      return projected && /^\{return ([\s\S]*);}$/.exec(projected)?.[1];
    }
    const call = completeCall(value);
    const definition = call && this.#recordFunctions.get(call.name);
    if (!call || !definition || this.#functionEffects.get(call.name) !== false) return undefined;
    if (fields.some((field) => !Object.hasOwn(definition.schema.propTypes, field)))
      return undefined;
    if (!unwrap && fields.length === Object.keys(definition.schema.propTypes).length)
      return undefined;
    const key = `${call.name}:${unwrap}:${fields.join(',')}`;
    let name = this.#recordProjections.get(key);
    if (!name) {
      const body = this.#functionBodies.get(call.name)!;
      const projected = projectRecordBody(
        body,
        fields,
        unwrap,
        (field) => {
          const type = this.emitTypeAnnotation(definition.schema.propTypes[field]!);
          return type.startsWith('cpuType') ? `${type}()` : type === 'bool' ? 'false' : '0';
        },
        (expression) => this.#projectRecord(expression, fields, unwrap),
      );
      if (!projected) return undefined;
      const simplified = prunePureLocals(projected, this.#functionEffects);
      name = this.ctx.makeUniqueIdentifier('cpuRecordFields', 'global');
      this.#recordProjections.set(key, name);
      this.#functions.add(name);
      this.#functionEffects.set(name, false);
      this.#functionBodies.set(name, simplified);
      this.#parameterUse.set(
        name,
        definition.parameters.map((parameter) =>
          new Set(simplified.match(/[A-Za-z_$][\w$]*/g)).has(parameter),
        ),
      );
      this.#recordArguments.set(
        name,
        definition.parameters.map((parameter) => usedRecordFields(simplified, parameter)),
      );
      this.ctx.addDeclaration(
        `function ${name}(${definition.parameters.join(',')}) ${simplified}`,
        name,
      );
    }
    return `${name}(${this.#projectArguments(name, call.arguments).join(',')})`;
  }
  #uniformDependencies(name: string, seen = new Set<string>()): Set<string> | undefined {
    if (seen.has(name) || this.#functionEffects.get(name) !== false) return undefined;
    const body = this.#functionBodies.get(name);
    if (!body) return undefined;
    seen.add(name);
    const tokens = new Set(body.match(/[A-Za-z_$][\w$]*/g));
    const dependencies = new Set<string>();
    for (const [id, global] of this.#globalScopes) {
      if (!tokens.has(id)) continue;
      if (global.scope === 'uniform') continue;
      if (global.scope === 'private' && ['u32', 'i32', 'f32', 'bool'].includes(global.type))
        dependencies.add(id);
      else return undefined;
    }
    for (const call of body.matchAll(/\b([A-Za-z_$][\w$]*)\(/g)) {
      if (!this.#functions.has(call[1]!)) continue;
      const nested = this.#uniformDependencies(call[1]!, new Set(seen));
      if (!nested) return undefined;
      for (const dependency of nested) dependencies.add(dependency);
    }
    return dependencies;
  }
  #cachedUniformArgument(value: string, fixedUniform = false): string {
    const call = /^([A-Za-z_$][\w$]*)\(\)$/.exec(value);
    if (!call && !fixedUniform) return value;
    const known = this.#uniformCaches.get(value);
    if (known) return `${known}()`;
    const dependencies = fixedUniform ? new Set<string>() : this.#uniformDependencies(call![1]!);
    if (!dependencies) return value;
    const cached = this.ctx.makeUniqueIdentifier('cpuUniformValue', 'global');
    const getter = this.ctx.makeUniqueIdentifier('cpuUniformRead', 'global');
    const keys = [...dependencies].map(() =>
      this.ctx.makeUniqueIdentifier('cpuUniformKey', 'global'),
    );
    this.ctx.addDeclaration(
      `//#dispatch-cache ${JSON.stringify([cached, ...keys])}\nlet ${[cached, ...keys].join(',')};\nfunction ${getter}(){if(${cached}===undefined${[...dependencies].map((dependency, i) => `||${keys[i]}!==${dependency}`).join('')}){${keys.map((key, i) => `${key}=${[...dependencies][i]};`).join('')}${cached}=${value};}return ${cached};}`,
      getter,
    );
    this.#uniformCaches.set(value, getter);
    return `${getter}()`;
  }
  #vectorReturns = new Map<
    string,
    { schema: d.BaseData; parameters: string[]; components: string[] }
  >();
  #recordReturns = new Map<string, { parameters: string[]; expression: string }>();
  #inlineRecord(code: string): { prefix: string; value: string } | undefined {
    const call = /^([A-Za-z_$][\w$]*)\(([\s\S]*)\)$/.exec(code);
    const record = call && this.#recordReturns.get(call[1]!);
    if (!record) return undefined;
    const values = callArguments(code);
    if (values.length !== record.parameters.length) return undefined;
    const names = values.map(() => this.ctx.makeUniqueIdentifier('cpuRecordInput', 'global'));
    const replacements = new Map(record.parameters.map((parameter, i) => [parameter, names[i]!]));
    const value = record.expression.replace(
      /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_$][\w$]*/g,
      (token, at: number) =>
        record.expression[at - 1] === '.' ? token : (replacements.get(token) ?? token),
    );
    const prefix = values.map((input, i) => `let ${names[i]} = ${input};\n`).join('');
    const nested = this.#inlineRecord(value);
    return nested ? { prefix: prefix + nested.prefix, value: nested.value } : { prefix, value };
  }
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
  #expressionLocals: string[] = [];
  #lowerPlaces(body: string): string {
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
    body = body.replace(/\bcpuVector(?:_\d+)?\b/g, (name) => {
      const parts = this.#components(name);
      return parts ? `$vector([${parts.join(',')}])` : name;
    });
    body = body.replace(/\bcpuAddress(?:_\d+)?\b/g, (name) => {
      const address = this.#addresses.get(name);
      return address ? readAddress(address) : name;
    });
    return body;
  }
  #scope(emit: () => ResolvedStatement): ResolvedStatement {
    const addresses = this.#addresses;
    const vectors = this.#vectors;
    const owned = this.#ownedLocals;
    this.#addresses = new Map(addresses);
    this.#vectors = new Map(vectors);
    this.#ownedLocals = new Set(owned);
    try {
      const result = emit();
      return { ...result, code: this.#lowerPlaces(result.code) };
    } finally {
      this.#addresses = addresses;
      this.#vectors = vectors;
      this.#ownedLocals = owned;
      for (const [name, address] of this.#globalAddresses) this.#addresses.set(name, address);
    }
  }
  #forgetLocal(name: string): void {
    this.#addresses.delete(name);
    this.#vectors.delete(name);
    this.#ownedLocals.delete(name);
  }
  protected override _block(
    block: Block,
    allowInlining: boolean,
    externalMap?: Record<string, unknown>,
  ): ResolvedStatement {
    // Lower references while their lexical bindings still exist.
    // eslint-disable-next-line no-underscore-dangle
    return this.#scope(() => super._block(block, allowInlining, externalMap));
  }
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
    if (!address || address.layout.type !== type)
      return type === 'f32' && /^[-+\d.e]+$/i.test(code) ? f32(code) : code;
    const value = readAddress(address);
    if (
      this.#globalScopes.get(address.base)?.scope === 'uniform' &&
      /^[\d()+*/% -]+$/.test(address.offset)
    )
      this.#uniformValues.add(value);
    return value;
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
  #normalizers = new Map<number, string>();

  protected override _expression(expression: Expression): Snippet {
    // Preserve integer overflow for unary negation before subsequent conversions.
    // eslint-disable-next-line no-underscore-dangle
    let result = super._expression(expression);
    if (
      Array.isArray(expression) &&
      expression[0] === NODE.conditionalExpr &&
      result.origin !== 'constant'
    ) {
      const call = completeCall(this.ctx.resolveSnippet(result).value);
      if (call?.name === 'select' && call.arguments.length === 3) {
        const [alternative, consequent, condition] = call.arguments;
        result = snip(
          this.emitCall(
            'select',
            [],
            [
              snip(alternative!, result.dataType, 'runtime', false),
              snip(consequent!, result.dataType, 'runtime', false),
              snip(condition!, d.bool, 'runtime', result.possibleSideEffects),
            ],
          ),
          result.dataType,
          result.origin,
          result.possibleSideEffects,
        );
      }
    }
    if (
      Array.isArray(expression) &&
      expression[0] === NODE.memberAccess &&
      typeof result.dataType !== 'symbol' &&
      result.origin !== 'constant'
    ) {
      const code = this.ctx.resolveSnippet(result).value;
      const property = /^(.*)\.([A-Za-z_$][\w$]*)$/.exec(code);
      const projected = property && this.#projectRecord(property[1]!, [property[2]!], true);
      if (projected)
        result = snip(projected, result.dataType, result.origin, result.possibleSideEffects);
    }
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

    if (
      typeof result.dataType !== 'symbol' &&
      ['mutable', 'readonly', 'uniform'].includes(result.origin)
    ) {
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
        `((v)=>$vector([${Array.from({ length: count }, (_, i) => (scalar(result.dataType).endsWith('i') ? `(-v[${i}]|0)` : `-v[${i}]`)).join(',')}]))(${this.#value(argument)})`,
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
    if (ctor === 'f32') return `${match[1]}${f32(value)};`;
    return `${match[1]}${ctor.startsWith('cpuType') && !address && !this.#ownedLocals.has(value) && !this.#fresh(value) ? `${ctor}(${value})` : value};`;
  }

  protected override _statement(statement: Statement): ResolvedStatement {
    if (Array.isArray(statement) && (statement[0] === NODE.for || statement[0] === NODE.forOf)) {
      // eslint-disable-next-line no-underscore-dangle
      return this.#scope(() => super._statement(statement));
    }
    if (Array.isArray(statement) && statement[0] === NODE.switch) {
      // TypeGPU emits case statements without calling _block; give each case an explicit block.
      // eslint-disable-next-line no-underscore-dangle
      return super._statement([
        NODE.switch,
        statement[1],
        (statement as Switch)[2].map(([test, body]) => [
          test,
          body.length ? [[NODE.block, [...body]]] : [],
        ]),
      ]);
    }
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
      this.#forgetLocal(name);
      this.#addresses.set(name, { ...address, offset: name });
      this.ctx.defineVariable(rawId, snip(name, value.dataType as d.BaseData, value.origin, false));
      return { code: `const ${name} = ${address.offset};`, definesInNearestScope: true };
    }
    const components = this.#parts(value);
    if (components) {
      const name = this.ctx.makeUniqueIdentifier(rawId, 'block');
      this.#forgetLocal(name);
      this.#vectors.set(name, components);
      this.ctx.defineVariable(rawId, snip(name, value.dataType as d.BaseData, value.origin, false));
      return { code: '', definesInNearestScope: false };
    }
    const type = scalar(value.dataType);
    if (type === 'ptr') return unsupported('implicit pointer alias');
    const name = this.ctx.makeUniqueIdentifier(rawId, 'block');
    this.#forgetLocal(name);
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
      const emitted = this.emitCall(name, [], args);
      return snip(
        typeof result.dataType !== 'symbol' &&
          result.dataType.type === 'f32' &&
          !EXACT_FLOAT_BUILTINS.has(name)
          ? f32(emitted)
          : emitted,
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
    return snip(
      scalar(schema) === 'f32' ? f32(String(value)) : String(value),
      schema,
      'constant',
      false,
    );
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
        type.endsWith('u')
          ? `((${value})>>>0)`
          : type.endsWith('i')
            ? `((${value})|0)`
            : f32(value);
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
          `$vector([${Array.from({ length: rows }, (_row, r) => f32(`(args[${c * rows + r}]??0)`)).join(',')}])`,
      ).join(',');
      const copied = Array.from(
        { length: columns },
        (_, c) =>
          `$vector([${Array.from({ length: rows }, (_row, r) => f32(`(columns[${c}]?.[${r}]??0)`)).join(',')}])`,
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
              : f32(`(${input}??0)`);
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
                : f32(`(${input}??0)`);
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
      const types = args.flatMap((arg) =>
        Array.from({ length: this.#parts(arg)?.length ?? 1 }, () => scalar(arg.dataType)),
      );
      const count = Number(schema.type[3]);
      const components = Array.from({ length: count }, (_, i) => {
        const value = parts.length === 1 ? parts[0]! : (parts[i] ?? '0');
        return schema.type.endsWith('u')
          ? integerCast(value, types[parts.length === 1 ? 0 : i] ?? '', true)
          : schema.type.endsWith('i')
            ? integerCast(value, types[parts.length === 1 ? 0 : i] ?? '', false)
            : /^(f32|vec[234]f)$/.test(types[parts.length === 1 ? 0 : i] ?? '')
              ? value
              : f32(value);
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
      const componentTypes = args.flatMap((arg) =>
        Array.from({ length: Number(/^vec([234])/.exec(scalar(arg.dataType))?.[1] ?? 1) }, () =>
          scalar(arg.dataType),
        ),
      );
      args.forEach((arg, index) => {
        const width = /^vec([234])/.exec(scalar(arg.dataType));
        if (width) for (let i = 0; i < Number(width[1]); i++) components.push(`a${index}[${i}]`);
        else components.push(`a${index}`);
      });
      const values = Array.from({ length: count }, (_, i) => {
        const value = components.length === 1 ? components[0]! : (components[i] ?? '0');
        return vectorType[2] === 'u'
          ? integerCast(value, componentTypes[components.length === 1 ? 0 : i] ?? '', true)
          : vectorType[2] === 'i'
            ? integerCast(value, componentTypes[components.length === 1 ? 0 : i] ?? '', false)
            : f32(value);
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
        ? integerCast(value, args.length ? scalar(args[0]!.dataType) : '', true)
        : type === 'i32'
          ? integerCast(value, args.length ? scalar(args[0]!.dataType) : '', false)
          : type === 'bool'
            ? `Boolean(${value})`
            : args.length && scalar(args[0]!.dataType) === 'f32'
              ? value
              : f32(value);
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
    const incomingAddress = this.#addresses.get(rhs);
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
    this.#forgetLocal(name);
    if (components) {
      const locals = components.map(() => this.ctx.makeUniqueIdentifier('cpuComponent', 'global'));
      this.#vectors.set(name, locals);
      return prefix + locals.map((local, i) => `let ${local}=${components[i]};`).join('');
    }
    const record = schema.type === 'struct' ? this.#inlineRecord(rhs) : undefined;
    if (record) {
      this.#ownedLocals.add(name);
      return `${record.prefix}let ${name} = ${record.value};`;
    }
    const type = this.emitTypeAnnotation(schema);
    const address = incomingAddress;
    if (address) rhs = readAddress(address);
    if (type === 'f32') return `let ${name} = ${f32(rhs)};`;
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
    for (const name of [id, `${id}_u32`, `${id}_i32`, `${id}_f32`])
      this.#globalScopes.set(name, { scope, type: scalar(dataType) });
    if (scope === 'handle') {
      const kind = dataType.type.startsWith('sampler')
        ? 'sampler'
        : dataType.type.startsWith('texture_storage')
          ? 'storageTexture'
          : 'texture';
      if (
        kind !== 'sampler' &&
        !['1d', '2d', '2d-array', '3d', 'cube', 'cube-array'].includes(
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
      let initial = options.init
        ? this.#value(options.init)
        : ctor.startsWith('cpuType')
          ? `${ctor}()`
          : '0';
      let reset: string | undefined;
      if (scope === 'workgroup' && !options.init && d.isWgslArray(dataType)) {
        const element = d.isAtomic(dataType.elementType)
          ? dataType.elementType.inner
          : dataType.elementType;
        if (element.type === 'u32' || element.type === 'i32') {
          initial = `new ${element.type === 'u32' ? 'Uint32Array' : 'Int32Array'}(${dataType.elementCount})`;
          reset = `${id}.fill(0);`;
        }
      }
      if (scope === 'private') this.#privateResets.push(`${id}=${initial};`);
      const scalarType = ['u32', 'i32', 'f32', 'bool'].includes(scalar(dataType));
      const expression =
        scope === 'private'
          ? initial
          : `$shared(${JSON.stringify(id)},()=>(${scalarType ? `{value:${initial}}` : initial}))`;
      this.ctx.addDeclaration(
        `//#scope ${scope}\n${scope === 'private' ? `//#private ${id}\n` : `//#workgroup ${JSON.stringify({ name: id, initial: expression, ...(reset ? { reset } : {}) })}\n`}let ${id} = ${expression};`,
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
    const previousExpressionLocals = this.#expressionLocals;
    this.#expressionLocals = [];
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
      body = this.#lowerPlaces(body);
      if (this.#expressionLocals.length)
        body = body.replace('{', `{let ${this.#expressionLocals.join(',')};`);
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
      body = prunePureLocals(this.#projectLocalRecords(body), this.#functionEffects);
      this.#functionEffects.set(options.name, this.#effects);
      this.#functionBodies.set(options.name, body);
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
      let vectorBody = body;
      for (;;) {
        const alias =
          /^\{\s*const ([A-Za-z_$][\w$]*) = ([A-Za-z_$\d\s()+*/%<>&|^.-]{1,120});\s*/.exec(
            vectorBody,
          );
        if (!alias || /[A-Za-z_$][\w$]*\(/.test(alias[2]!)) break;
        vectorBody = `{${vectorBody.slice(alias[0].length)}`;
        const name = alias[1]!.replace(/[$]/g, '\\$');
        vectorBody = vectorBody.replace(
          new RegExp(`(?<![\\w$.])${name}(?![\\w$])`, 'g'),
          () => `(${alias[2]})`,
        );
      }
      const vectorReturn = /^\{\s*return \$vector\(\[([\s\S]*)\]\);\s*}$/.exec(vectorBody);
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
        const flattened = args.flatMap((arg) => parameters.get(arg.name) ?? [arg.name]);
        this.#recordArguments.set(
          options.name,
          args.flatMap((arg) =>
            parameters.has(arg.name)
              ? parameters.get(arg.name)!.map(() => undefined)
              : [
                  arg.decoratedType.type === 'struct'
                    ? usedRecordFields(body, arg.name)
                    : undefined,
                ],
          ),
        );
        if (!this.#effects && returnType.type === 'struct')
          this.#recordFunctions.set(options.name, {
            schema: returnType as d.WgslStruct,
            parameters: flattened,
          });
        const record = /^\{\s*return ([\s\S]*);\s*}$/.exec(body);
        if (!this.#effects && returnType.type === 'struct' && record) {
          const expression = record[1]!;
          const forwarded = /^([A-Za-z_$][\w$]*)\(/.exec(expression);
          if (expression.startsWith('({') || (forwarded && this.#recordReturns.has(forwarded[1]!)))
            this.#recordReturns.set(options.name, {
              parameters: args.flatMap((arg) => parameters.get(arg.name) ?? [arg.name]),
              expression,
            });
        }
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
      return `//#kit-gl compute\n//#kit-gl reusable\n//#entry ${JSON.stringify({ name: options.name, size, inputs, generator, ...(phases ? { phases: phases.count } : {}) })}\nfunction${generator ? '*' : ''} ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}${phases ? `\nconst ${options.name}_phases=${phases.source};` : ''}`;
    } finally {
      this.#expressionLocals = previousExpressionLocals;
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
    if (op === '=' || (op.endsWith('=') && !['==', '!=', '<=', '>='].includes(op))) {
      const indexed = /^(\w+)\[([\s\S]+)\]$/.exec(this.ctx.resolveSnippet(lhs).value);
      const components = indexed && this.#components(indexed[1]!);
      if (components) {
        const index = this.ctx.makeUniqueIdentifier('cpuIndex', 'global');
        const value = this.ctx.makeUniqueIdentifier('cpuAssigned', 'global');
        const cases = components.map((component, i) => {
          const assigned = this.emitBinaryOp(
            snip(component, lhs.dataType as d.BaseData, lhs.origin, false),
            op,
            snip(value, rhs.dataType as d.BaseData, 'runtime', false),
          );
          return `case ${i}:return (${assigned});`;
        });
        // Scalarized vectors have no array to mutate. Evaluate the index and RHS once, then
        // assign the selected component rather than a reconstructed vector temporary.
        return `((${index},${value})=>{switch(${index}){${cases.join('')}}})(${indexed![2]},${this.#value(rhs)})`;
      }
    }
    const leftParts = this.#parts(lhs),
      rightParts = this.#parts(rhs);
    const vector = type.startsWith('vec') ? lhs : rhsType.startsWith('vec') ? rhs : undefined;
    const width = vector ? Number(scalar(vector.dataType)[3]) : 0;
    const targetAddress = this.#addresses.get(this.ctx.resolveSnippet(lhs).value);
    if (vector && leftParts && ['+=', '-=', '*=', '/=', '%='].includes(op)) {
      if (
        !rightParts &&
        rhsType.startsWith('vec') &&
        leftParts.every((part) => /^[A-Za-z_$][\w$]*$/.test(part))
      ) {
        const primitive = type.endsWith('u') ? d.u32 : type.endsWith('i') ? d.i32 : d.f32;
        const parameters = leftParts.map((_, i) => `a${i}`);
        const results = leftParts.map((_, i) =>
          this.emitBinaryOp(
            snip(parameters[i]!, primitive, 'runtime', false),
            op.slice(0, -1) as BinaryOperator,
            snip(`b[${i}]`, primitive, 'runtime', false),
          ),
        );
        return `((${[...parameters, 'b'].join(',')})=>{const ${results.map((value, i) => `v${i}=${value}`).join(',')};${leftParts.map((part, i) => `${part}=v${i};`).join('')}})(${[...leftParts, this.#value(rhs)].join(',')})`;
      }
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
            f32dot(
              matrixLeft.map((column) => column[r]!),
              rightParts,
            ),
          ),
        );
      }
      if (op === '*' && matrixRight && leftParts && !lhs.possibleSideEffects)
        return this.#vector(
          d[`vec${matrixRight.length}f` as 'vec2f'],
          matrixRight.map((column) => f32dot(leftParts, column)),
        );
      let output: string;
      if (op === '*' && lm && rhsType.startsWith('vec'))
        output = `$vector([${Array.from({ length: Number(lm[2]) }, (_r, r) =>
          f32dot(
            Array.from({ length: Number(lm[1]) }, (_c, c) => `a[${c}][${r}]`),
            Array.from({ length: Number(lm[1]) }, (_c, c) => `b[${c}]`),
          ),
        ).join(',')}])`;
      else if (op === '*' && rm && type.startsWith('vec'))
        output = `$vector([${Array.from({ length: Number(rm[1]) }, (_c, c) =>
          f32dot(
            Array.from({ length: Number(rm[2]) }, (_r, r) => `a[${r}]`),
            Array.from({ length: Number(rm[2]) }, (_r, r) => `b[${c}][${r}]`),
          ),
        ).join(',')}])`;
      else if (op === '*' && lm && rm)
        output = `[${Array.from(
          { length: Number(rm[1]) },
          (_c, c) =>
            `$vector([${Array.from({ length: Number(lm[2]) }, (_r, r) =>
              f32dot(
                Array.from({ length: Number(lm[1]) }, (_k, k) => `a[${k}][${r}]`),
                Array.from({ length: Number(lm[1]) }, (_k, k) => `b[${c}][${k}]`),
              ),
            ).join(',')}])`,
        ).join(',')}]`;
      else if (['+', '-', '*', '/'].includes(op)) {
        const shape = lm ?? rm!;
        output = `[${Array.from({ length: Number(shape[1]) }, (_c, c) => `$vector([${Array.from({ length: Number(shape[2]) }, (_r, r) => f32op(lm ? `a[${c}][${r}]` : 'a', op, rm ? `b[${c}][${r}]` : 'b')).join(',')}])`).join(',')}]`;
      } else return unsupported(`matrix operator ${op}`);
      const inputs = new Map<string, string>();
      output = output.replace(/\b[ab]\[\d\](?:\[\d\])?/g, (access) => {
        let local = inputs.get(access);
        if (!local) {
          local = `v${inputs.size}`;
          inputs.set(access, local);
        }
        return local;
      });
      const locals = [...inputs]
        .map(([access, local]) => `const ${local}=${f32(access)};`)
        .join('');
      return `((a,b)=>{${locals}return ${output};})(${left},${right})`;
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
    if (op.endsWith('=') && !['=', '==', '!=', '<=', '>='].includes(op)) {
      const place = addressParts(this.#value(lhs));
      const simple = /^[A-Za-z_$][\w$]*$|^\d+$|^"[^"\\]*"$/;
      if (place && (!simple.test(place.object) || !simple.test(place.key))) {
        const index = this.ctx.makeUniqueIdentifier('cpuIndex', 'global');
        const base = simple.test(place.object)
          ? place.object
          : this.ctx.makeUniqueIdentifier('cpuBase', 'global');
        const assignment = this.emitBinaryOp(
          snip(`${base}[${index}]`, lhs.dataType as d.BaseData, 'runtime'),
          op,
          rhs,
        );
        // Keep potentially suspending RHS calls in the enclosing generator frame.
        this.#expressionLocals.push(index);
        if (base !== place.object) this.#expressionLocals.push(base);
        return `(${base === place.object ? '' : `${base}=${place.object},`}${index}=${place.key},${assignment})`;
      }
    }
    const float = type === 'f32' || rhsType === 'f32';
    const left = `(${float ? f32Literal(this.#value(lhs)) : this.#value(lhs)})`;
    const right = `(${float ? f32Literal(this.#value(rhs)) : this.#value(rhs)})`;
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
    if (float && ['+', '-', '*', '/', '%'].includes(operator))
      result = f32op(this.#value(lhs), operator, this.#value(rhs));
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
      const to = match[2]!.toLowerCase();
      const source = scalar(args[0]!.dataType);
      const address = this.#addresses.get(this.ctx.resolveSnippet(args[0]!).value);
      if (source.startsWith('vec')) {
        const count = Number(source[3]);
        const schema = d[`vec${count}${to[0]}` as 'vec2f'];
        if (address)
          return this.#vector(
            schema,
            Array.from({ length: count }, (_, i) =>
              readAddress({
                ...address,
                offset: `(${address.offset})+${i * 4}`,
                layout: { type: to, size: 4 },
              }),
            ),
          );
        const components = this.#parts(args[0]!);
        if (components && !args[0]!.possibleSideEffects)
          return this.#vector(
            schema,
            components.map((value) => `$bits.${match[1]!.toLowerCase()}_${to}(${value})`),
          );
        return `((v)=>$vector([${Array.from({ length: count }, (_, i) => `$bits.${match[1]!.toLowerCase()}_${to}(v[${i}])`).join(',')}]))(${this.#value(args[0]!)})`;
      }
      if (address) return readAddress({ ...address, layout: { type: to, size: 4 } });
      return `$bits.${match[1]!.toLowerCase()}_${to}(${this.#value(args[0]!)})`;
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
      const address = this.#addresses.get(this.ctx.resolveSnippet(args[0]!).value);
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
        if (address)
          return this.#vector(
            targetSchema,
            Array.from({ length: count }, (_, i) =>
              readAddress({
                base: address.base,
                offset: `(${address.offset})+${i * 4}`,
                layout: { type: to, size: 4 },
              }),
            ),
          );
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
      if (address) return readAddress({ ...address, layout: { type: target, size: 4 } });
      return `$bits.${source}_${target}(${this.#value(args[0]!)})`;
    }
    if (templates.length) unsupported(`template call ${name}`);
    const floatArguments = args.some((arg) => /^(f32|vec[234]f)$/.test(scalar(arg.dataType)));
    const values = args.map((arg) => {
      const value = this.#value(arg);
      return floatArguments && scalar(arg.dataType).startsWith('abstract') ? f32(value) : value;
    });
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
            (value) =>
              /^[\w$.[\]()+\-*/%<>&|^ ]+$/.test(value) &&
              !/[\w$]\(/.test(value.replaceAll('Math.fround(', '(')),
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
        let value = type === 'f32' ? f32(values[i]!) : values[i]!;
        const parameter = widths.slice(0, i).reduce<number>((sum, width) => sum + (width ?? 1), 0);
        const fields = this.#recordArguments.get(name)?.[parameter];
        if (fields?.length && type === 'struct')
          value = this.#projectRecord(value, fields, false) ?? value;
        if (!widths[i] && (['struct', 'array'].includes(type) || type.startsWith('mat'))) {
          // Parameters are immutable values; only this borrowed-argument path can reuse a record.
          if (type === 'struct' || type.startsWith('mat')) {
            const cached = this.#cachedUniformArgument(value, this.#uniformValues.has(value));
            if (cached !== value) return cached;
          }
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
      const primitive = schema.type.endsWith('u')
        ? d.u32
        : schema.type.endsWith('i')
          ? d.i32
          : d.f32;
      const direct = args.map((arg) => this.#parts(arg));
      const canInline = args.every(
        (arg, i) =>
          !arg.possibleSideEffects && (!scalar(arg.dataType).startsWith('vec') || direct[i]),
      );
      if (name === 'normalize') {
        let helper = this.#normalizers.get(count);
        if (!helper) {
          helper = this.ctx.makeUniqueIdentifier('cpuNormalize', 'global');
          this.#normalizers.set(count, helper);
          this.#functions.add(helper);
          this.#functionEffects.set(helper, false);
          const parameters = Array.from({ length: count }, (_, i) => `v${i}`);
          const local = parameters.map((_, i) => `x${i}`);
          const declarations = local
            .map((value, i) => `const ${value}=${f32(parameters[i]!)};`)
            .join('');
          this.ctx.addDeclaration(
            `function ${helper}(${parameters.join(',')}){${declarations}const length=${f32(`Math.sqrt(${f32dot(local, local)})`)};return $vector([${local.map((value) => f32op(value, '/', 'length')).join(',')}]);}`,
            helper,
          );
        }
        return canInline
          ? `${helper}(${direct[0]!.join(',')})`
          : `((v)=>${helper}(${Array.from({ length: count }, (_, i) => `v[${i}]`).join(',')}))(${values[0]})`;
      }
      const letters = args.map((_, i) => `a${i}`);
      const at = (index: number, i: number) =>
        canInline
          ? `(${direct[index]?.[i] ?? this.#value(args[index]!)})`
          : scalar(args[index]!.dataType).startsWith('vec')
            ? `${letters[index]}[${i}]`
            : letters[index]!;
      const components = (index: number) => Array.from({ length: count }, (_, i) => at(index, i));
      const operation = (left: string, op: BinaryOperator, right: string) =>
        this.emitBinaryOp(
          snip(left, primitive, 'runtime', false),
          op,
          snip(right, primitive, 'runtime', false),
        );
      const dot = (a: string[], b: string[]) =>
        a
          .map((value, i) => operation(value, '*', b[i]!))
          .reduce((sum, value) => operation(sum, '+', value));
      const product = dot(
        components(0),
        components(['dot', 'reflect', 'refract'].includes(name) ? 1 : 0),
      );
      let result: string | undefined;
      let parts: string[] | undefined;
      if (name === 'dot') result = product;
      else if (name === 'length') result = f32(`Math.sqrt(${product})`);
      else if (name === 'distance') {
        const difference = components(0).map((value, i) => f32op(value, '-', at(1, i)));
        result = f32(`Math.sqrt(${f32dot(difference, difference)})`);
      } else if (name === 'reflect')
        parts = components(0).map((value, i) =>
          f32op(value, '-', f32op(f32op('2', '*', product), '*', at(1, i))),
        );
      else if (name === 'refract') {
        const eta = at(2, 0);
        const k = f32op(
          '1',
          '-',
          f32op(f32op(eta, '*', eta), '*', f32op('1', '-', f32op(product, '*', product))),
        );
        parts = components(0).map(
          (value, i) =>
            `((k)=>k<0?0:${f32op(f32op(eta, '*', value), '-', f32op(f32op(f32op(eta, '*', product), '+', f32('Math.sqrt(k)')), '*', at(1, i)))} )(${k})`,
        );
      } else if (name === 'cross')
        parts = [0, 1, 2].map((i) =>
          f32op(
            f32op(at(0, (i + 1) % 3), '*', at(1, (i + 2) % 3)),
            '-',
            f32op(at(0, (i + 2) % 3), '*', at(1, (i + 1) % 3)),
          ),
        );
      else
        parts = Array.from({ length: count }, (_, i) =>
          this.emitCall(
            name,
            [],
            args.map((arg, j) => {
              const type = scalar(arg.dataType);
              const componentType = type.startsWith('vec')
                ? type.endsWith('u')
                  ? d.u32
                  : type.endsWith('i')
                    ? d.i32
                    : type.endsWith('b')
                      ? d.bool
                      : d.f32
                : (arg.dataType as d.BaseData);
              return snip(at(j, i), componentType, 'runtime', false);
            }),
          ),
        );
      if (canInline) return result ?? this.#vector(schema, parts!);
      return `((${letters.join(',')})=>${result ?? `$vector([${parts!.join(',')}])`})(${values.join(',')})`;
    }
    if (name === 'reverseBits')
      return `((a)=>{let r=0;for(let i=0;i<32;i++){r=(r<<1)|(a&1);a>>>=1;}return r${scalar(args[0]!.dataType) === 'i32' ? '|0' : '>>>0'};})(${values[0]})`;
    if (name === 'countOneBits')
      return `((a)=>{a=a-((a>>>1)&0x55555555);a=(a&0x33333333)+((a>>>2)&0x33333333);return Math.imul((a+(a>>>4))&0x0f0f0f0f,0x01010101)>>>24;})(${values[0]})`;
    const expressions: Record<string, string> = {
      not: '!a',
      smoothstep: `((t)=>${f32op(f32op('t', '*', 't'), '*', f32op('3', '-', f32op('2', '*', 't')))})(Math.min(Math.max(${f32op(f32op('c', '-', 'a'), '/', f32op('b', '-', 'a'))},0),1))`,
      clamp: 'Math.min(Math.max(a,b),c)',
      mix: f32op(f32op('a', '*', f32op('1', '-', 'c')), '+', f32op('b', '*', 'c')),
      fract: f32op('a', '-', 'Math.floor(a)'),
      inverseSqrt: f32op('1', '/', f32('Math.sqrt(a)')),
      saturate: 'Math.min(Math.max(a,0),1)',
      step: 'b<a?0:1',
      fma: 'a*b+c',
      radians: f32op('a', '*', String(Math.PI / 180)),
      degrees: f32op('a', '*', String(180 / Math.PI)),
      countLeadingZeros: 'Math.clz32(a)',
      countTrailingZeros: 'a===0?32:31-Math.clz32((a&-a)>>>0)',
      firstTrailingBit: 'a===0?4294967295:31-Math.clz32((a&-a)>>>0)',
    };
    const floatResult = scalar(args[0]!.dataType) === 'f32';
    const rounded = (value: string) => {
      const type = scalar(args[0]!.dataType);
      if (type === 'i32' || type === 'u32') return integerCast(value, type, type === 'u32');
      return floatResult && !EXACT_FLOAT_BUILTINS.has(name) ? f32(value) : value;
    };
    if (expressions[name]) return `((a,b,c)=>(${rounded(expressions[name])}))(${values.join(',')})`;
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
      if (name === 'exp2') return rounded(`(2 ** (${values[0]}))`);
      if (name === 'round')
        return `((a)=>{const f=Math.floor(a);return a-f===0.5?(f%2===0?f:f+1):Math.round(a);})(${values[0]})`;
      return rounded(`Math.${name}(${values.join(',')})`);
    }
    if (name === 'select')
      return args.some((arg) => arg.possibleSideEffects)
        ? `((a,b,c)=>c?b:a)(${values.join(',')})`
        : `(${values[2]} ? ${values[1]} : ${values[0]})`;
    return unsupported(`builtin ${name}`);
  }
}
