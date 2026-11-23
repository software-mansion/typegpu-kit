import type { TgpuRoot } from 'typegpu';

/**
 * Throws unless `count` threads, `size` to a workgroup, fit in the workgroups the device dispatches
 * along one dimension, and names the largest count that fits.
 */
export function checkDispatch(
  root: TgpuRoot,
  where: string,
  what: string,
  count: number,
  size: number,
) {
  const most = root.device.limits.maxComputeWorkgroupsPerDimension;
  const groups = Math.ceil(count / size);
  if (groups <= most) return;
  throw new RangeError(
    `${where} needs ${groups} workgroups along x for a \`${what}\` of ${count}. The device's \`maxComputeWorkgroupsPerDimension\` is ${most}. ` +
      `The largest ${what} it allows is ${most * size}.`,
  );
}
