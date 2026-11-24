export function compileJavascript(
  parameters: string[],
  source: string,
): (...args: unknown[]) => unknown {
  try {
    return new Function(...parameters, source) as (...args: unknown[]) => unknown;
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    const lines = source.split('\n');
    // Function constructors omit syntax-error locations in several JS engines. Removing one
    // source line identifies localized emission errors without adding a parser to the runtime.
    for (let index = 0; index < lines.length; index++) {
      if (!lines[index]!.trim() || lines[index]!.trimStart().startsWith('//')) continue;
      const candidate = [...lines];
      candidate[index] = '';
      try {
        Function(...parameters, candidate.join('\n'));
      } catch {
        continue;
      }
      throw new SyntaxError(
        `@typegpu/kit-gl CPU: ${error.message}\nGenerated line ${index + 1}: ${lines[index]}`,
        { cause: error },
      );
    }
    const numbered = lines.map((line, index) => `${index + 1}: ${line}`).join('\n');
    throw new SyntaxError(
      `@typegpu/kit-gl CPU: ${error.message}\nCould not isolate one line; generated source:\n${numbered}`,
      { cause: error },
    );
  }
}
