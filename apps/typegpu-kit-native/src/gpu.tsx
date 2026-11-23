import * as native from '@typegpu/kit/native';
import * as runtimes from '@typegpu/kit/native/runtimes';
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { examples } from 'typegpu-kit-examples';
import { colors } from './theme.ts';

/** Features the examples use when the adapter has them, and the debug panel's timestamps. */
const FEATURES = [
  ...new Set([...examples.flatMap((example) => example.features ?? []), 'timestamp-query']),
] as GPUFeatureName[];

interface Gpu {
  readonly device: GPUDevice;
  /** The worker every example runs on, so scenes opened in a session share its modules. */
  readonly worker: runtimes.WorkerRuntime;
}

const GpuContext = createContext<Gpu | undefined>(undefined);

/** Returns the app's device and worker, which every example shares. */
export function useGpu(): Gpu {
  const gpu = useContext(GpuContext);
  if (!gpu) throw new Error('useGpu needs a <GpuGate> above it.');
  return gpu;
}

async function requestDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter.');
  // react-native-webgpu adds implicit-device-synchronization to every device it requests
  const requiredFeatures = FEATURES.filter((feature) => adapter.features.has(feature));
  return native.checkDevice(await adapter.requestDevice({ requiredFeatures }));
}

/**
 * Requests the app's device and worker, and shows its children once they are ready. A lost device
 * is replaced, so an example restarted after the loss runs on a live one.
 */
export function GpuGate({ children }: { readonly children: ReactNode }) {
  const [worker] = useState(() => runtimes.worker({ name: 'Examples' }));
  const [device, setDevice] = useState<GPUDevice>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    let live = true;
    const start = () => {
      void requestDevice().then(
        (next) => {
          if (!live) return;
          setDevice(next);
          void next.lost.then((info) => {
            if (live && info.reason !== 'destroyed') start();
          });
        },
        (reason: unknown) => {
          if (!live) return;
          setDevice(undefined);
          setError(String(reason));
        },
      );
    };
    start();
    return () => {
      live = false;
    };
  }, []);

  const gpu = useMemo(() => device && { device, worker }, [device, worker]);

  if (!gpu) {
    return (
      <View style={styles.gate}>
        <Text style={[styles.text, error !== undefined && styles.error]}>
          {error ?? 'Starting the GPU…'}
        </Text>
      </View>
    );
  }

  return <GpuContext.Provider value={gpu}>{children}</GpuContext.Provider>;
}

const styles = StyleSheet.create({
  gate: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    backgroundColor: colors.ground,
  },
  text: { color: colors.muted, fontSize: 15, textAlign: 'center' },
  error: { color: colors.exception },
});
