/** Error scopes and `uncapturederror` events, as WebGPU reports errors. */

export type ErrorFilter = 'validation' | 'out-of-memory' | 'internal';

export interface Errors {
  readonly target: EventTarget;
  report(filter: ErrorFilter, message: string): void;
  push(filter: GPUErrorFilter): void;
  pop(): Promise<GPUError | null>;
}

type ErrorConstructor = new (message: string) => GPUError;

const constructors: Record<ErrorFilter, string> = {
  validation: 'GPUValidationError',
  'out-of-memory': 'GPUOutOfMemoryError',
  internal: 'GPUInternalError',
};

/** A `GPUError` of the filter's kind: WebGPU's own class where the browser has it. */
export function gpuError(filter: ErrorFilter, message: string): GPUError {
  const name = constructors[filter];
  const Native = (globalThis as unknown as Record<string, ErrorConstructor | undefined>)[name];
  if (Native) return new Native(message);
  const error = new Error(message);
  error.name = name;
  return error as unknown as GPUError;
}

function uncapturedEvent(error: GPUError): Event {
  const Native = (
    globalThis as unknown as {
      GPUUncapturedErrorEvent?: new (type: string, init: { error: GPUError }) => Event;
    }
  ).GPUUncapturedErrorEvent;
  if (Native) return new Native('uncapturederror', { error });
  return Object.assign(new Event('uncapturederror'), { error });
}

export function createErrors(onUncaptured: () => ((event: Event) => void) | null): Errors {
  const target = new EventTarget();
  const scopes: { filter: GPUErrorFilter; error: GPUError | null }[] = [];

  return {
    target,

    report(filter, message) {
      for (let i = scopes.length - 1; i >= 0; i--) {
        const scope = scopes[i]!;
        if (scope.filter !== filter) continue;
        scope.error ??= gpuError(filter, message);
        return;
      }
      const event = uncapturedEvent(gpuError(filter, message));
      target.dispatchEvent(event);
      onUncaptured()?.(event);
    },

    push(filter) {
      scopes.push({ filter, error: null });
    },

    pop() {
      const scope = scopes.pop();
      if (!scope)
        return Promise.reject(new DOMException('No error scope to pop.', 'OperationError'));
      return Promise.resolve(scope.error);
    },
  };
}
