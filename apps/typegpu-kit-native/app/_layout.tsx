import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GpuGate } from '../src/gpu.tsx';
import { colors, DISPLAY } from '../src/theme.ts';

export default function Layout() {
  const [loaded] = useFonts({
    [DISPLAY]: require('@typegpu/kit/fonts/InterDisplay-SemiBold.ttf') as number,
  });
  if (!loaded) return null;

  return (
    <SafeAreaProvider>
      <GestureHandlerRootView style={styles.fill}>
        <StatusBar style="light" />
        <GpuGate>
          <Stack
            screenOptions={{
              headerShown: false,
              contentStyle: { backgroundColor: colors.ground },
            }}
          />
        </GpuGate>
      </GestureHandlerRootView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: colors.ground },
});
