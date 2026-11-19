import { renderIndex } from './agents.ts';
import { extractContracts } from './extract.ts';

const started = performance.now();
const { contracts, issues } = extractContracts();
const seconds = ((performance.now() - started) / 1000).toFixed(2);

console.log(renderIndex(contracts));

const undefaulted = contracts.flatMap((contract) =>
  contract.options.flatMap((table) =>
    table.fields
      .filter((field) => field.optional && !field.default)
      .map((field) => `${contract.id}: ${table.name}.${field.name}`),
  ),
);
if (undefaulted.length > 0) {
  console.log(`\n${undefaulted.length} optional fields say no default ("Default …."):`);
  for (const each of undefaulted) console.log(`- ${each}`);
}

console.log(`\n${contracts.length} modules in ${seconds} s.`);
if (issues.length > 0) {
  console.log(`\n${issues.length} issues:`);
  for (const issue of issues) console.log(`- ${issue}`);
  process.exitCode = 1;
}
