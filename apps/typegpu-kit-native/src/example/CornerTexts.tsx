import { StyleSheet, Text, View, type ViewStyle } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { Readings } from 'typegpu-kit-examples/hosts';
import type { Corner, Ui } from 'typegpu-kit-examples/ui.ts';
import { colors, fonts } from '../theme.ts';
import { EDGE, elementsOf, UNDER_BAR } from './overlay.ts';

interface CornerTextsProps {
  readonly ui: Ui | undefined;
  /** The scene's current texts, by element name. */
  readonly texts: Readings['texts'] | undefined;
}

/** The example's text elements at their corners, such as a game's score. */
export function CornerTexts({ ui, texts }: CornerTextsProps) {
  return (
    <>
      {elementsOf(ui, 'text').map(([name, element]) => {
        const text = texts?.[name];
        return text ? <CornerText key={name} corner={element.corner} text={text} /> : null;
      })}
    </>
  );
}

const CORNERS: Readonly<Record<Corner, ViewStyle>> = {
  'top-left': { top: 0, left: 0, alignItems: 'flex-start' },
  'top-right': { top: 0, right: 0, alignItems: 'flex-end' },
  'bottom-left': { bottom: 0, left: 0, alignItems: 'flex-start' },
  'bottom-right': { bottom: 0, right: 0, alignItems: 'flex-end' },
};

function CornerText({ corner, text }: { readonly corner: Corner; readonly text: string }) {
  const insets = useSafeAreaInsets();
  const top = corner.startsWith('top');
  return (
    <View
      pointerEvents="none"
      style={[
        styles.corner,
        CORNERS[corner],
        top ? { marginTop: insets.top + UNDER_BAR } : { marginBottom: insets.bottom + EDGE },
      ]}
    >
      <Text style={[styles.text, { textAlign: corner.endsWith('right') ? 'right' : 'left' }]}>
        {text}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  corner: { position: 'absolute', marginHorizontal: EDGE },
  text: {
    ...fonts.mono,
    color: colors.text,
    fontSize: 15,
    lineHeight: 21,
    textShadowColor: 'rgba(0, 0, 0, 0.8)',
    textShadowRadius: 6,
  },
});
