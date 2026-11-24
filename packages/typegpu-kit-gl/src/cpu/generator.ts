import { layoutOf } from './memory.ts';
import { NodeTypeCatalog as NODE, type Expression, type Statement } from 'tinyest';
import { d } from 'typegpu';
import {
  WgslGenerator,
  snip,
  type BinaryOperator,
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
    if (type === 'ptr') return unsupported('pointer alias');
    const name = this.ctx.makeUniqueIdentifier(rawId, 'block');
    this.ctx.defineVariable(rawId, snip(name, value.dataType as d.BaseData, value.origin, false));
    return {
      code: `const ${name} = ${this.ctx.resolveSnippet(value).value};`,
      definesInNearestScope: true,
    };
  }

  override numericLiteral(value: number, schema: d.BaseData): ResolvedSnippet {
    return snip(String(value), schema, 'constant', false);
  }

  override emitTypeAnnotation(schema: d.BaseData): string {
    const type = scalar(schema);
    if (['u32', 'i32', 'f32', 'bool', 'void', 'abstractInt', 'abstractFloat'].includes(type)) {
      return type;
    }
    return unsupported(`type ${type}`);
  }

  override typeInstantiation(schema: d.BaseData, args: readonly Snippet[]): ResolvedSnippet {
    const type = this.emitTypeAnnotation(schema);
    const value = args.length ? this.ctx.resolveSnippet(args[0]!).value : '0';
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
    this.emitTypeAnnotation(schema);
    return `let ${name} = ${rhs};`;
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
    if (!['uniform', 'readonly', 'mutable'].includes(scope)) unsupported(`${scope} variable ${id}`);
    const array = dataType.type === 'array' ? (dataType as d.WgslArray) : undefined;
    const type = scalar(array?.elementType ?? dataType);
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
    const body = this._block(options.body, false).code;
    if (/\b(workgroupBarrier|storageBarrier|textureBarrier)\s*\(/.test(body))
      unsupported('builtin workgroupBarrier/storageBarrier/textureBarrier');
    const returnType = options.determineReturnType();
    this.emitTypeAnnotation(returnType);
    const args = options.args.filter((arg) => arg.used || options.functionType === 'normal');
    if (options.functionType === 'normal') {
      for (const arg of args) this.emitTypeAnnotation(arg.decoratedType);
      return `function ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}`;
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
    return `//#kit-gl compute\n//#entry ${JSON.stringify({ name: options.name, size, inputs })}\nfunction ${options.name}(${args.map((arg) => arg.name).join(',')}) ${body || '{}'}`;
  }

  override emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
    const type = scalar(lhs.dataType);
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
    if (templates.length) unsupported(`template call ${name}`);
    const values = args.map((arg) => this.ctx.resolveSnippet(arg).value);
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
