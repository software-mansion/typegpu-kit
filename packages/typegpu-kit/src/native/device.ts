import { useMainDevice } from 'react-native-webgpu';

/** Returns the device of the enclosing `GPUDeviceProvider`, or undefined outside one. */
export function useProvidedDevice(): GPUDevice | undefined {
  try {
    return useMainDevice().device ?? undefined;
  } catch {
    return undefined; // react-native-webgpu throws outside a provider
  }
}
