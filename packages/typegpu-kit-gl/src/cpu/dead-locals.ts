/** Drop unused declarations only when their generated initializer has no observable effects. */
export function prunePureLocals(body: string, effects: ReadonlyMap<string, boolean>): string {
  const pure = (expression: string) => {
    if (/=>|;|\+\+|--|\byield\b|(?<![<>=!])=(?!=)/.test(expression)) return false;
    for (const call of expression.matchAll(/([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/g)) {
      const name = call[1]!;
      if (
        name === '$vector' ||
        name === 'Number' ||
        name === 'Boolean' ||
        /^Math\.[A-Za-z]+$/.test(name) ||
        /^cpuType(?:_\d+)?(?:\.fields)?$/.test(name) ||
        effects.get(name) === false
      )
        continue;
      return false;
    }
    return true;
  };
  for (;;) {
    let changed = false;
    for (const declaration of body.matchAll(/^let ([A-Za-z_$][\w$]*) = ([^\n]*);$/gm)) {
      if (!pure(declaration[2]!)) continue;
      const name = declaration[1]!;
      const names = body.match(/[A-Za-z_$][\w$]*/g) ?? [];
      if (names.filter((token) => token === name).length !== 1) continue;
      body = body.replace(declaration[0], '');
      changed = true;
    }
    if (!changed) return body;
  }
}
