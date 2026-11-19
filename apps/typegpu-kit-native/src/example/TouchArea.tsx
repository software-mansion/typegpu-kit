import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AppState, StyleSheet, View, type HostInstance } from 'react-native';
import {
  GestureDetector,
  GestureStateManager,
  useManualGesture,
  type GestureTouchEvent,
  type TouchData,
} from 'react-native-gesture-handler';
import { useSharedValue } from 'react-native-reanimated';
import { scheduleOnUI } from 'react-native-worklets';
import { Canvas } from 'react-native-webgpu';
import type { Example } from 'typegpu-kit-examples';
import type { PadRect, TouchLayout } from '../touches.ts';
import * as touch from '../touch.ts';
import { elementsOf } from './overlay.ts';
import { Pads } from './Pads.tsx';
import type { ExampleHandle } from './useExample.ts';

interface TouchAreaProps {
  readonly example: Example;
  readonly hosted: ExampleHandle;
}

/** The last lane id given to a touch area. */
let lastLane = 0;

/**
 * The canvas and the pads over it, under one touch lane (touch.ts). Each finger is routed once,
 * where it lands, so pads, a drag and a pinch run together. The top bar, the debug panel and the
 * settings sheet sit outside it.
 */
export function TouchArea({ example, hosted }: TouchAreaProps) {
  const [lane] = useState(() => ++lastLane);
  const area = useRef<HostInstance>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [rects, setRects] = useState<Readonly<Record<string, PadRect>>>({});
  const lit = useSharedValue<Readonly<Record<string, boolean>>>({});
  const foregrounds = useForegrounds();

  useEffect(() => () => scheduleOnUI(touch.forget, lane), [lane]);

  const { shape } = hosted;
  const send = hosted.scene?.send;
  const callbacks = useMemo(() => {
    const layout: TouchLayout = {
      width: size.width,
      height: size.height,
      shape,
      pads: Object.values(rects),
    };
    const target: touch.Target = { send, lit };
    const deliver = (kind: touch.TouchKind, changed: readonly TouchData[]) => {
      'worklet';
      const flat: number[] = [];
      for (const { id, x, y } of changed) flat.push(id, x, y);
      touch.route(lane, kind, flat, performance.now(), target, layout);
    };
    return {
      onTouchesDown: (event: GestureTouchEvent) => {
        'worklet';
        deliver('down', event.changedTouches);
      },
      onTouchesMove: (event: GestureTouchEvent) => {
        'worklet';
        deliver('move', event.changedTouches);
      },
      onTouchesUp: (event: GestureTouchEvent) => {
        'worklet';
        deliver('up', event.changedTouches);
        if (event.numberOfTouches === 0) GestureStateManager.fail(event.handlerTag);
      },
      onTouchesCancel: (event: GestureTouchEvent) => {
        'worklet';
        deliver('cancel', event.changedTouches);
        if (event.numberOfTouches === 0) GestureStateManager.fail(event.handlerTag);
      },
      onFinalize: () => {
        'worklet';
        touch.settle(lane);
      },
    };
  }, [lane, size, shape, rects, send, lit]);
  const gesture = useManualGesture(callbacks);

  const place = useCallback(
    (rect: PadRect) =>
      setRects((all) => ({ ...all, [touch.padKey(rect.element, rect.name)]: rect })),
    [],
  );

  return (
    <View style={StyleSheet.absoluteFill}>
      {/* Outside the remounted detector, so a return to the foreground keeps the running scene */}
      <Canvas {...hosted.canvas} style={StyleSheet.absoluteFill} />
      {/* iOS cancels touches on the way to the background, which can leave a Manual recognizer
          stuck, so each return to the foreground makes a new one */}
      <GestureDetector key={foregrounds} gesture={gesture}>
        <View
          ref={area}
          style={StyleSheet.absoluteFill}
          onLayout={(event) => setSize(event.nativeEvent.layout)}
        >
          {elementsOf(example.ui, 'buttons').map(([name, buttons]) => (
            <Pads
              key={name}
              element={name}
              buttons={buttons}
              area={area}
              lit={lit}
              onPlace={place}
            />
          ))}
        </View>
      </GestureDetector>
    </View>
  );
}

/** Counts returns to the foreground. */
function useForegrounds() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state === 'active') setCount((each) => each + 1);
    });
    return () => subscription.remove();
  }, []);
  return count;
}
