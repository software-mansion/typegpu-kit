import { Stack, useLocalSearchParams } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { examples } from 'typegpu-kit-examples';
import { ExampleScreen } from '../src/example/ExampleScreen.tsx';
import { colors } from '../src/theme.ts';

export default function ExampleRoute() {
  const { name } = useLocalSearchParams<{ name: string }>();
  const example = examples.find((each) => each.key === `${name}`);
  if (!example) {
    return (
      <View style={styles.missing}>
        <Text style={styles.text}>{`No example ${name}.`}</Text>
      </View>
    );
  }
  return (
    <>
      {/* Every drag on an example belongs to it, so leaving is the back button's alone */}
      <Stack.Screen options={{ gestureEnabled: false, fullScreenGestureEnabled: false }} />
      <ExampleScreen example={example} />
    </>
  );
}

const styles = StyleSheet.create({
  missing: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  text: { color: colors.muted, fontSize: 15 },
});
