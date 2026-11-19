import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { examples, highlights, type Example } from 'typegpu-kit-examples';
import { categories, type Category } from '../src/catalog.ts';
import { draw } from 'typegpu-kit-pull/rarity.ts';
import type { Placed } from 'typegpu-kit-pull/timeline.ts';
import { entries } from '../src/chest/cards.ts';
import { Chest } from '../src/chest/Chest.tsx';
import { Pull, type Opened } from '../src/chest/Pull.tsx';
import { Banner } from '../src/chrome/Banner.tsx';
import { posterOf } from '../src/files.ts';
import { colors, fonts } from '../src/theme.ts';

/** Points between the screen's edges and the cards. */
const MARGIN = 20;
/** Points between cards in a row. */
const GAP = 14;
/** Screens at least this many points wide show three columns, narrower ones two. */
const WIDE = 700;
/** Points of the next highlight that show beside the one in view. */
const PEEK = 36;

/** The gallery: a dithered banner with the site's box on it over each example's poster by category. */
export default function Gallery() {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const columns = width >= WIDE ? 3 : 2;
  const cardWidth = (width - 2 * MARGIN - (columns - 1) * GAP) / columns;
  const focused = useFocused();
  const [bannerHeight, setBannerHeight] = useState(0);
  const [scrolledPast, setScrolledPast] = useState(false);
  const [pull, setPull] = useState<Opened | null>(null);
  const [away, setAway] = useState(false);

  const open = useCallback((from: Placed) => {
    const { item, direction, phase } = draw(entries);
    setPull({ entry: item, from, direction, phase });
  }, []);
  const close = useCallback(() => {
    setPull(null);
    setAway(false);
  }, []);
  const shown = useCallback(() => setAway(true), []);
  const pick = useCallback(
    (key: string) => {
      close();
      router.push(`/${key}`);
    },
    [close],
  );

  return (
    <View style={styles.fill}>
      <ScrollView
        style={styles.fill}
        contentContainerStyle={{ paddingBottom: insets.bottom + 40 }}
        scrollEventThrottle={100}
        onScroll={(event) => {
          const past = event.nativeEvent.contentOffset.y > bannerHeight;
          if (past !== scrolledPast) setScrolledPast(past);
        }}
      >
        <View onLayout={(event) => setBannerHeight(event.nativeEvent.layout.height)}>
          <Banner paused={!focused || scrolledPast}>
            <View style={[styles.head, { paddingTop: insets.top + 28 }]}>
              <Text style={styles.kicker}>TYPEGPU KIT</Text>
              <Text style={styles.heading}>Examples</Text>
              <Text style={styles.lede}>
                {`${examples.length} scenes from the docs, each running on this device's GPU with the same code as the site.`}
              </Text>
              <View style={styles.rest}>
                <Text style={styles.restLabel}>{'OPEN THE BOX\nFOR A RANDOM EXAMPLE'}</Text>
                <Chest paused={!focused || scrolledPast} away={away} onOpen={open} />
              </View>
            </View>
          </Banner>
        </View>
        <Highlights width={width} />
        <View style={styles.body}>
          {categories.map((category) => (
            <Section key={category.name} category={category} cardWidth={cardWidth} />
          ))}
        </View>
      </ScrollView>
      <Pull pull={pull} onShown={shown} onPick={pick} onClose={close} />
    </View>
  );
}

/** The highlights as large cards in a row that snaps to each, the next one peeking in. */
function Highlights({ width }: { readonly width: number }) {
  const cardWidth = Math.min(width - 2 * MARGIN - PEEK, 560);
  return (
    <View style={styles.section}>
      <View style={[styles.sectionHead, styles.body]}>
        <Text style={styles.sectionTitle}>HIGHLIGHTS</Text>
        <View style={styles.rule} />
        <Text style={styles.count}>{String(highlights.length).padStart(2, '0')}</Text>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        decelerationRate="fast"
        snapToInterval={cardWidth + GAP}
        contentContainerStyle={styles.reel}
      >
        {highlights.map((example) => (
          <Card key={example.key} example={example} width={cardWidth} large />
        ))}
      </ScrollView>
    </View>
  );
}

/** Whether the screen is on top of the stack. */
function useFocused() {
  const [focused, setFocused] = useState(true);
  useFocusEffect(
    useCallback(() => {
      setFocused(true);
      return () => setFocused(false);
    }, []),
  );
  return focused;
}

interface SectionProps {
  readonly category: Category;
  readonly cardWidth: number;
}

/** A category's heading, ruled out to its count, over its cards. */
function Section({ category, cardWidth }: SectionProps) {
  return (
    <View style={styles.section}>
      <View style={styles.sectionHead}>
        <Text style={styles.sectionTitle}>{category.title.toUpperCase()}</Text>
        <View style={styles.rule} />
        <Text style={styles.count}>{String(category.examples.length).padStart(2, '0')}</Text>
      </View>
      <View style={styles.grid}>
        {category.examples.map((example) => (
          <Card key={example.key} example={example} width={cardWidth} />
        ))}
      </View>
    </View>
  );
}

interface CardProps {
  readonly example: Example;
  readonly width: number;
  /** Sets the title and line larger, for a highlight. */
  readonly large?: boolean;
}

/** An example's poster, title and line, opening the example when pressed. */
function Card({ example, width, large = false }: CardProps) {
  const poster = posterOf(example.key);
  return (
    <Pressable
      onPress={() => router.push(`/${example.key}`)}
      accessibilityRole="button"
      accessibilityLabel={example.title}
      style={({ pressed }) => [{ width }, styles.card, pressed && styles.pressed]}
    >
      <View style={styles.frame}>
        {poster !== undefined ? <Image source={poster} style={styles.poster} /> : null}
      </View>
      <Text style={[styles.title, large && styles.titleLarge]} numberOfLines={1}>
        {example.title}
      </Text>
      <Text style={[styles.line, large && styles.lineLarge]} numberOfLines={2}>
        {example.line}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: colors.ground },
  head: { paddingHorizontal: MARGIN, paddingBottom: 28 },
  body: { paddingHorizontal: MARGIN },
  kicker: { ...fonts.mono, color: colors.link, fontSize: 12, letterSpacing: 2.4 },
  heading: { ...fonts.display, color: colors.text, fontSize: 40, marginTop: 6 },
  lede: { color: colors.muted, fontSize: 15, lineHeight: 21, marginTop: 10, maxWidth: 520 },
  rest: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  restLabel: {
    ...fonts.mono,
    color: colors.muted,
    fontSize: 11,
    lineHeight: 17,
    letterSpacing: 1.6,
    marginBottom: 14,
  },
  reel: { paddingHorizontal: MARGIN, gap: GAP },
  section: { marginTop: 36 },
  sectionHead: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 14 },
  sectionTitle: { ...fonts.mono, color: colors.text2, fontSize: 12, letterSpacing: 2 },
  rule: { flex: 1, height: StyleSheet.hairlineWidth, backgroundColor: colors.ruleStrong },
  count: { ...fonts.mono, color: colors.muted, fontSize: 12 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: GAP, rowGap: 22 },
  card: { gap: 4 },
  pressed: { opacity: 0.72, transform: [{ scale: 0.98 }] },
  frame: {
    aspectRatio: 16 / 10,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: colors.ground3,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.ruleStrong,
    marginBottom: 6,
  },
  poster: { width: '100%', height: '100%' },
  title: { ...fonts.display, color: colors.text, fontSize: 16 },
  titleLarge: { fontSize: 20 },
  line: { color: colors.muted, fontSize: 13, lineHeight: 18 },
  lineLarge: { fontSize: 14, lineHeight: 20 },
});
