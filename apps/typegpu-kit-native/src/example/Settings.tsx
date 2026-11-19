import {
  BottomSheet,
  Button,
  Form,
  Group,
  HStack,
  Host,
  LabeledContent,
  Picker,
  Section,
  Slider,
  Spacer,
  Text,
  Toggle,
  VStack,
} from '@expo/ui/swift-ui';
import {
  disabled as disable,
  foregroundStyle,
  monospacedDigit,
  pickerStyle,
  presentationBackgroundInteraction,
  presentationDetents,
  presentationDragIndicator,
  tag,
} from '@expo/ui/swift-ui/modifiers';
import { useEffect, useState } from 'react';
import { StyleSheet } from 'react-native';
import type { Readings, Shape } from 'typegpu-kit-examples/hosts';
import type { Knob, Range, Value, Values } from 'typegpu-kit-examples/settings.ts';
import type { Ui } from 'typegpu-kit-examples/ui.ts';
import { colors } from '../theme.ts';
import { elementsOf } from './overlay.ts';

/** Choices up to this many show as segments, more as a menu. */
const SEGMENTS = 4;

const SECONDARY = { type: 'hierarchical', style: 'secondary' } as const;

export interface SettingsProps {
  readonly title: string;
  readonly open: boolean;
  readonly onOpenChange: (open: boolean) => void;
  readonly settings: Shape['settings'];
  readonly ui: Ui | undefined;
  readonly readings: Readings | undefined;
  readonly write: (key: string, value: Value) => void;
  readonly act: (name: string) => void;
  readonly pick: (element: string) => void;
  /** The camera element streaming, if any. */
  readonly filming: string | undefined;
  /** Turns camera element `element` on, or off when it streams. */
  readonly film: (element: string) => void;
}

/** The example's settings as native controls in a sheet over its canvas. */
export function Settings({
  title,
  open,
  onOpenChange,
  settings,
  ui,
  readings,
  write,
  act,
  pick,
  filming,
  film,
}: SettingsProps) {
  const [values, setValues] = useState<Values>(settings?.values ?? {});

  // Live knobs follow the scene, such as the hex isle clock
  const live = readings?.values;
  useEffect(() => {
    if (!settings || !live) return;
    setValues((shown) => {
      const next = { ...shown };
      for (const [key, knob] of Object.entries(settings.knobs)) {
        if (knob.live && live[key] !== undefined) next[key] = live[key];
      }
      return next;
    });
  }, [settings, live]);

  const change = (key: string, value: Value) => {
    setValues((shown) => ({ ...shown, [key]: value }));
    write(key, value);
  };

  const knobs = Object.entries(settings?.knobs ?? {});
  const folders = [...new Set(knobs.map(([, knob]) => knob.folder))];
  const images = elementsOf(ui, 'image');
  const cameras = elementsOf(ui, 'camera');
  const actions = settings?.actions ?? [];
  const readouts = Object.entries(readings?.readouts ?? {});

  return (
    <Host style={styles.host} colorScheme="dark">
      <BottomSheet isPresented={open} onIsPresentedChange={onOpenChange}>
        <Group
          modifiers={[
            presentationDetents([{ fraction: 0.4 }, 'large']),
            presentationBackgroundInteraction({
              type: 'enabledUpThrough',
              detent: { fraction: 0.4 },
            }),
            presentationDragIndicator('visible'),
          ]}
        >
          <Form>
            {folders.map((folder) => (
              <Section key={folder ?? ''} title={folder ?? title}>
                {knobs
                  .filter(([, knob]) => knob.folder === folder)
                  .map(([key, knob]) => (
                    <Control
                      key={key}
                      name={key}
                      knob={knob}
                      value={values[key]}
                      onChange={(value) => change(key, value)}
                    />
                  ))}
              </Section>
            ))}
            {actions.length + images.length + cameras.length > 0 ? (
              <Section title="actions">
                {actions.map((name) => (
                  <Button key={name} label={name} onPress={() => act(name)} />
                ))}
                {images.map(([name, image]) => (
                  <Button key={name} label={image.label} onPress={() => pick(name)} />
                ))}
                {cameras.map(([name, camera]) => (
                  <Button
                    key={name}
                    label={filming === name ? 'stop camera' : camera.label}
                    onPress={() => film(name)}
                  />
                ))}
              </Section>
            ) : null}
            {readouts.length > 0 ? (
              <Section title="readouts">
                {readouts.map(([name, value]) => (
                  <LabeledContent key={name} label={name}>
                    <Text modifiers={[monospacedDigit(), foregroundStyle(SECONDARY)]}>
                      {format(value)}
                    </Text>
                  </LabeledContent>
                ))}
              </Section>
            ) : null}
          </Form>
        </Group>
      </BottomSheet>
    </Host>
  );
}

interface ControlProps {
  readonly name: string;
  readonly knob: Knob;
  readonly value: Value | undefined;
  readonly onChange: (value: Value) => void;
}

function Control({ name, knob, value, onChange }: ControlProps) {
  const label = knob.label ?? name;
  const off = knob.disabled === undefined ? [] : [disable()];
  const reason = knob.disabled ? (
    <Text modifiers={[foregroundStyle(SECONDARY)]}>{knob.disabled}</Text>
  ) : null;

  if (typeof value === 'boolean') {
    return (
      <VStack alignment="leading" spacing={2}>
        <Toggle label={label} isOn={value} onIsOnChange={onChange} modifiers={off} />
        {reason}
      </VStack>
    );
  }

  if (typeof value !== 'boolean' && value !== undefined && 'options' in knob) {
    const options = knob.options as readonly (string | number)[];
    return (
      <VStack alignment="leading" spacing={2}>
        <Picker
          label={label}
          selection={value}
          onSelectionChange={onChange}
          modifiers={[pickerStyle(options.length <= SEGMENTS ? 'segmented' : 'menu'), ...off]}
        >
          {options.map((option) => (
            <Text key={option} modifiers={[tag(option)]}>
              {format(option)}
            </Text>
          ))}
        </Picker>
        {reason}
      </VStack>
    );
  }

  if (typeof value === 'number' && 'min' in knob) {
    const { min, max, step } = knob as Range;
    return (
      <VStack alignment="leading" spacing={4}>
        <HStack>
          <Text>{label}</Text>
          <Spacer />
          <Text modifiers={[monospacedDigit(), foregroundStyle(SECONDARY)]}>{format(value)}</Text>
        </HStack>
        <Slider
          value={value}
          min={min}
          max={max}
          step={step}
          onValueChange={onChange}
          modifiers={off}
        />
        {reason}
      </VStack>
    );
  }

  return null;
}

/** Up to three decimals, without trailing zeros. */
const format = (value: Value) =>
  typeof value === 'number' ? String(Math.round(value * 1000) / 1000) : String(value);

const styles = StyleSheet.create({
  // The sheet presents itself; its host takes no room
  host: { position: 'absolute', width: 0, height: 0, backgroundColor: colors.ground },
});
