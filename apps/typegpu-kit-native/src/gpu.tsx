import * as native from '@typegpu/kit/native';
import * as runtimes from '@typegpu/kit/native/runtimes';
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors } from './theme.ts';

/** Features the examples use when the adapter has them. */
const FEATURES = [
  'indirect-first-instance',
  'timestamp-query',
  'texture-formats-tier1',
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

async function requestGpu(): Promise<Gpu> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter.');
  // react-native-webgpu adds implicit-device-synchronization to every device it requests
  const requiredFeatures = FEATURES.filter((feature) => adapter.features.has(feature));
  const device = await adapter.requestDevice({ requiredFeatures });
  native.checkDevice(device);
  return { device, worker: runtimes.worker({ name: 'Examples' }) };
}

/** Requests the app's device and worker, and shows its children once they are ready. */
export function GpuGate({ children }: { readonly children: ReactNode }) {
  const [gpu, setGpu] = useState<Gpu>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    requestGpu().then(setGpu, (reason: unknown) => setError(String(reason)));
  }, []);

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
