/** Split only unconditional entry barriers; helpers and control-flow barriers keep generators. */
export function splitPhases(
  body: string,
  parameters: string[],
): { source: string; count: number } | undefined {
  const phases: string[] = [];
  const locals = new Set(parameters);
  let code = '';
  let braces = 0;
  let parentheses = 0;
  let brackets = 0;
  let quote = '';
  let comment = '';
  const functions: boolean[] = [];
  let returned = false;
  for (let i = 0; i < body.length; i++) {
    const char = body[i]!;
    const next = body[i + 1];
    if (comment) {
      code += char;
      if (comment === 'line' && char === '\n') comment = '';
      else if (comment === 'block' && char === '*' && next === '/') {
        code += next;
        i++;
        comment = '';
      }
      continue;
    }
    if (quote) {
      code += char;
      if (char === '\\') code += body[++i];
      else if (char === quote) quote = '';
      continue;
    }
    if (char === '"' || char === "'") quote = char;
    if (char === '`') return undefined;
    if (char === '/' && (next === '/' || next === '*')) {
      comment = next === '/' ? 'line' : 'block';
      code += char + next;
      i++;
      continue;
    }
    if (body.startsWith('(yield)', i)) {
      if (braces !== 1 || parentheses !== 0 || brackets !== 0 || returned) return undefined;
      phases.push(code);
      code = '';
      i += '(yield)'.length - 1;
      if (body[i + 1] === ';') i++;
      continue;
    }
    if (/^yield\b/.test(body.slice(i))) return undefined;
    if (braces === 1 && parentheses === 0 && brackets === 0) {
      const declaration = /^(let|const)\s+([A-Za-z_$][\w$]*)\s*=/.exec(body.slice(i));
      if (declaration) {
        locals.add(declaration[2]!);
        code += `${declaration[2]}=`;
        i += declaration[0].length - 1;
        continue;
      }
    }
    if (/^return\b/.test(body.slice(i)) && !functions.some(Boolean)) returned = true;
    if (char === '{') {
      functions.push(/=>\s*$|function[^{}]*\)\s*$/.test(body.slice(0, i)));
      braces++;
      if (braces === 1) continue;
    }
    if (char === '}') {
      functions.pop();
      braces--;
      if (braces === 0) continue;
    }
    if (char === '(') parentheses++;
    if (char === ')') parentheses--;
    if (char === '[') brackets++;
    if (char === ']') brackets--;
    code += char;
  }
  if (!phases.length) return undefined;
  phases.push(code);
  const inputs = parameters.map((_, i) => `cpuInput${i}`);
  const initialize = parameters.map((name, i) => `${name}=${inputs[i]};`).join('');
  return {
    count: phases.length,
    source: `function(){${locals.size ? `let ${[...locals].join(',')};` : ''}return [${phases.map((phase, i) => `function(${i === 0 ? inputs.join(',') : ''}){${i === 0 ? initialize : ''}${phase}}`).join(',')}];}`,
  };
}
