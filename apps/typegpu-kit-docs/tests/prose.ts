import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { satteri } from '@astrojs/markdown-satteri';
import { repo, type Extraction } from '../src/contracts/extract.ts';

export async function inlineCodeOf(file: string) {
  const spans: { code: string; line: number }[] = [];
  const processor = satteri({
    features: { directive: true },
    mdastPlugins: [
      {
        name: 'inline-code-spans',
        options: { position: true },
        inlineCode(node) {
          spans.push({ code: node.value, line: node.position!.start.line });
        },
      },
    ],
  });
  const source = readFileSync(file, 'utf8');
  const options = { syntaxHighlight: false } as const;
  if (file.endsWith('.mdx')) {
    const renderer = await processor.createMdxRenderer!(options, { optimize: false });
    await renderer.process(source, file, {});
  } else {
    const renderer = await processor.createRenderer(options);
    await renderer.render(source, { fileURL: pathToFileURL(file) });
  }
  return spans;
}

export function namespaceMembers(
  { contracts, program }: Extraction,
  { allowHandleMembers = false } = {},
) {
  const checker = program.getTypeChecker();
  return new Map(
    contracts
      .filter((contract) => !contract.flat)
      .map((contract) => {
        // The cards list options and returned types separately and may omit hidden exports
        const source = program.getSourceFile(join(repo.package, contract.file))!;
        const symbol = checker.getSymbolAtLocation(source)!;
        const members = checker.getExportsOfModule(symbol).map((member) => member.name);
        if (allowHandleMembers) {
          for (const handle of contract.handles) {
            for (const member of handle.members) members.push(member.name);
          }
        }
        return [contract.namespace, new Set(members)] as const;
      }),
  );
}

export function unknownMembers(code: string, namespaces: ReadonlyMap<string, ReadonlySet<string>>) {
  // A path segment or a property on a longer receiver does not name a namespace
  return [...code.matchAll(/(?<![\w$./-])([a-z]\w*)\.([a-zA-Z]\w*)/g)].flatMap(
    ([ref, name, member]) => {
      const members = namespaces.get(name!);
      return members && !members.has(member!) ? [ref] : [];
    },
  );
}
