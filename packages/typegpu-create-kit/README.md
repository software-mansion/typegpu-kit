# @typegpu/create-kit

Starts a Vite project with [TypeGPU Kit](https://docs.swmansion.com/typegpu-kit/) and a scene to build on:

```sh
npm create @typegpu/kit my-scene
```

The project is the one [Getting started](https://docs.swmansion.com/typegpu-kit/getting-started/) builds: Vite with the TypeGPU plugin, tsover, the kit and a scene hosted on a canvas. Its `AGENTS.md` points coding agents at the kit's skill. The directory must be new or empty.

In a terminal it asks which of the [starter scenes](https://docs.swmansion.com/typegpu-kit/starter-scenes/) to begin with. `--scene` names one without asking, and without a terminal it writes `shader`:

```sh
npm create @typegpu/kit my-scene -- --scene lit
```

| Scene        | Draws                                               |
| ------------ | --------------------------------------------------- |
| `shader`     | one fragment function over the whole canvas         |
| `lit`        | a mesh, a sun and its shadow, under an orbit camera |
| `finished`   | the lit scene with HDR, bloom and a tone curve      |
| `simulation` | particles a compute kernel moves each step          |

## License

MIT
