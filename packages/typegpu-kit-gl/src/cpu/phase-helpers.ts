import { callArguments } from './references.ts';

export interface BarrierHelper {
  body: string;
  parameters: string[];
}

// Only scalar helpers with straight-line barriers can be expanded. Control-flow barriers,
// closures and nested suspension keep the general generator scheduler.
export function expandBarrierHelpers(
  body: string,
  helpers: Map<string, BarrierHelper>,
  unique: () => string,
): string {
  return body.replace(
    /\blet (\w+)\s*=\s*\(yield\* (\w+)\(([^;\n]*)\)\);/g,
    (original: string, result: string, name: string, argumentsCode: string) => {
      const helper = helpers.get(name);
      if (
        !helper ||
        /=>|\b(function|for|while|switch|do|break|continue)\b|yield\*/.test(helper.body)
      )
        return original;
      const tail = /\breturn ([^;]+);\s*}$/.exec(helper.body);
      if (!tail || [...helper.body.matchAll(/\breturn\b/g)].length !== 1) return original;
      const values = callArguments(`call(${argumentsCode})`);
      if (values.length !== helper.parameters.length) return original;
      const parameterNames = helper.parameters.map(unique);
      const scopes: Map<string, string>[] = [
        new Map(helper.parameters.map((parameter, i) => [parameter, parameterNames[i]!])),
      ];
      const braces: boolean[] = [];
      let declaration = false;
      let previous = '';
      let code = '';
      const tokens =
        helper.body.match(
          /\/\/[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_$][\w$]*|\s+|./g,
        ) ?? [];
      for (const token of tokens) {
        if (/^\s|^\/\//.test(token) || token.startsWith('/*')) {
          code += token;
          continue;
        }
        if (token === '{') {
          const transparent = braces.length === 0 || ['{', '}', ';'].includes(previous);
          braces.push(transparent);
          scopes.push(new Map());
          if (!transparent) code += token;
        } else if (token === '}') {
          if (!braces.pop()) code += token;
          scopes.pop();
        } else if (token === 'let' || token === 'const') {
          declaration = true;
          code += token;
        } else if (/^[A-Za-z_$][\w$]*$/.test(token)) {
          if (declaration) {
            scopes.at(-1)!.set(token, unique());
            declaration = false;
          }
          let replacement = token;
          if (previous !== '.')
            for (let i = scopes.length - 1; i >= 0; i--) {
              const found = scopes[i]!.get(token);
              if (found) {
                replacement = found;
                break;
              }
            }
          code += replacement;
        } else code += token;
        previous = token;
      }
      code = code.replace(/\breturn ([^;]+);\s*$/, `${result}=$1;`);
      return `let ${result}=undefined;${parameterNames.map((parameter, i) => `let ${parameter}=${values[i]};`).join('')}${code}`;
    },
  );
}
