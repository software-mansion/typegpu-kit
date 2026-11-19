# The contract

These declarations change only by addition. A release can add an optional member, an overload or
a name, and removes or changes nothing here. `tests/contract.test.ts` prints them from the source
and fails on any difference. Accept an addition with
`pnpm vitest run --project='!*(gpu)' -u packages/typegpu-kit/tests/contract.test.ts`.

## frames.ts

```ts
export interface Frames extends Disposable {
    run(seconds: number): boolean;
    readonly seconds: number | undefined;
    destroy(): void;
}

export declare function create(root: TgpuRoot, record: Recorder): Frames;

export declare const encoderOf: (target: Frame | TgpuCommandEncoder) => TgpuCommandEncoder;

export declare const dtOf: (target: Frame | TgpuCommandEncoder, dt?: number) => number;

export interface Frame {
    readonly encoder: TgpuCommandEncoder;
    readonly time: number;
    readonly dt: number;
    readonly index: number;
    afterSubmit(callback: () => void): void;
    submit(): void;
}
```

## owner.ts

```ts
export interface Owner extends Disposable {
    <T extends Ownable | undefined>(item: T): T;
    readonly destroyed: boolean;
    readonly warmables: readonly Warmable[];
    usable(): void;
    destroy(): void;
}

export declare function owner(message?: string): Owner;
```

## inject.ts

```ts
export type Inject = (cfg: Configurable) => Configurable;

export interface Injectable {
    inject(): Inject;
}

export declare const byStage: <T extends d.AnyWgslData>(buffer: TgpuBuffer<T> & StorageFlag) => TgpuAccessor.In<T>;

export declare const unfilled: (what: string, instance: string, origin: string) => () => never;

export type Checked<O, S> = O extends S ? (O extends Known<O, S> ? O : Known<O, S>) : S;
```

## passes.ts

```ts
export interface Drawing<V extends VertexStruct = VertexStruct, P = never> extends Shaders<V, P> {
    inject?(): Inject;
    readonly indices?: TgpuBuffer<d.BaseData> & IndexFlag;
    draw(pass: TgpuRenderCommands): void;
    initAsync?(): Promise<void>;
}

export interface Compound<P = never, X = never> {
    readonly '~fed'?: (pixel: P, vertex: X) => void;
    build(into: Into): Built;
}

export interface Into<W extends Writes = Writes> extends Injectable {
    readonly writes: W;
    readonly depthFormat: GPUTextureFormat | undefined;
    readonly depthOnly: boolean;
    readonly depthCompare: GPUCompareFunction;
    readonly samples: number;
    readonly enabled: boolean;
    readonly layers: number;
    inject(): Inject;
    add<V extends VertexStruct, P>(drawing: Drawing<V, P> & Fits<W, P, d.Infer<V>>): Drawn;
    add<P, X>(drawing: Compound<P, X> & Fits<W, P, X>): Drawn;
    pipeline<V extends VertexStruct, P>(drawing: Shaders<V, P>, from?: WithBinding): TgpuRenderPipeline;
    without<K extends OutputKeys<W>>(...keys: K[]): Into<Omit<W, K>>;
    instead<const I extends {
        readonly [K in OutputKeys<W>]?: Output<never, never>;
    }>(outputs: I): Into<Omit<W, keyof I> & I>;
}

export interface Built {
    draw(pass: TgpuRenderCommands, layer: number): void;
    initAsync(): Promise<void>;
    destroy?(): void;
}

export interface Drawn extends Disposable {
    draw(pass: TgpuRenderCommands): void;
    initAsync(): Promise<void>;
    destroy(): void;
}

export interface Output<in P = unknown, in X = unknown> {
    readonly '~reads'?: (pixel: P, vertex: X) => void;
}

export declare function output<V extends Held, P = unknown, X = unknown, C extends Struct | undefined = undefined>(schema: V, spec: {
    readonly clear?: d.InferGPU<V>;
    readonly varyings?: C;
    readonly vary?: (current: X, previous: X) => Carried<C>;
    readonly previous?: boolean;
    readonly write: (pixel: P, carried: Carried<C>) => d.Infer<V>;
}): Output<P, X>;
```

## cameras.ts

```ts
export declare const current: import("typegpu").TgpuAccessor<d.WgslStruct<{
    viewProjection: d.Mat4x4f;
    position: d.Vec3f;
    right: d.Vec3f;
    up: d.Vec3f;
}>>;

export declare const Camera: d.WgslStruct<{
    viewProjection: d.Mat4x4f;
    position: d.Vec3f;
    right: d.Vec3f;
    up: d.Vec3f;
}>;
export type Camera = d.InferGPU<typeof Camera>;
```

## targets.ts

```ts
export type FloatTexture = TgpuTexture<{
    size: readonly number[];
    format: FloatFormat;
    sampleCount?: 1;
}> & SampledFlag;

export interface Surface extends SizeSource, Disposable {
    readonly format: GPUTextureFormat;
    readonly width: number;
    readonly height: number;
    readonly pixelRatio: number;
    view(frame: Frame): GPUTextureView;
    destroy(): void;
}
```

## post/post.ts

```ts
export interface Rendered {
    readonly output: FloatTexture;
    readonly region?: SizeSource | undefined;
}

export type Source = (() => FloatTexture) | Rendered;
```

## hosting.ts

```ts
export interface Scene {
    frame(frame: Frame): void;
    destroy?(): void;
    readonly ready?: PromiseLike<unknown>;
}

export type CreateScene<Options, S extends Scene> = (root: TgpuRoot, surface: Surface, options: Options) => S | Promise<S>;
```

## testing.ts

```ts
export interface RecordingLog {
    readonly shaders: string[];
    readonly events: string[];
    readonly state: string[];
    readonly writes: {
        buffer: GPUBuffer;
        offset: number;
        size: number;
    }[];
    buffers: number;
    textures: number;
    views: number;
    bindGroups: number;
    samplers: number;
    querySets: number;
    pipelines: number;
    blocking: number;
    submissions: number;
    counts(): RecordingCounts;
    alive(options?: AliveOptions): string[];
}

export interface RecordingCounts {
    readonly buffers: number;
    readonly textures: number;
    readonly views: number;
    readonly bindGroups: number;
    readonly samplers: number;
    readonly querySets: number;
    readonly pipelines: number;
}

export interface RecordingRoot {
    readonly root: TgpuRoot;
    readonly log: RecordingLog;
    bytes(buffer: GPUBuffer | TgpuBuffer<d.BaseData>): Uint8Array;
}

export interface RecordingOptions {
    readonly features?: readonly string[];
    readonly limits?: Partial<Record<keyof GPUSupportedLimits, number>>;
}

export declare function recordingRoot(options?: RecordingOptions): RecordingRoot;
```
