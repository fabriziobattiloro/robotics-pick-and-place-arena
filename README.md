# Franka Panda — Pick & Place with Vision-Language Models

A browser-based embodied-reasoning demo. A vision-language model (Gemini, Claude, or GPT) looks at a simulated 3D scene, returns 2D detections for the objects you asked about, and the Franka Emika Panda arm picks them up and drops them in a tray.

> This project started as a fork of Google's official AI Studio app — [Robotics: Franka Pick and Place](https://ai.studio/apps/bundled/robotics_franka_pick_and_place) — which was Gemini-only. It has been extended with support for **Anthropic Claude** and **OpenAI GPT** models, a **Run All Models** benchmarking mode, saved cube scenes, optimised per-provider prompts, and a two-pass detection pipeline.

## How it works

The app closes a **Sense → Plan → Act** loop every time you hit *Send*:

1. **Sense** — `RenderSystem` snapshots the current Three.js canvas as a JPEG.
2. **Plan** — the image + your text prompt go to the selected VLM, which returns JSON (2D boxes, points, or segmentation masks).
3. **Project** — each 2D detection is raycast into the 3D scene to get a world-space target.
4. **Act** — `SequenceAnimator` drives the arm through Hover → Open → Lower → Grasp → Lift → Move → Drop, using an analytical IK solver (`FrankaAnalyticalIK`) for the 7-DOF redundancy.

Stack: **React** (UI) + **Three.js** (rendering) + **MuJoCo WASM** (physics & collisions) + **Vercel AI SDK** (model calls). See `Code.md` for a file-by-file tour.

## Run locally

**Prerequisites:** Node.js 18+

```bash
npm install
cp .env.example .env.local   # then fill in the keys for providers you want
npm run dev
```

You only need keys for the providers you plan to test; missing keys disable the matching models in the UI.

## Running experiments

Everything lives in the right-hand sidebar.

1. **Load a scene** — use the scene selector to pick a saved cube environment (or interact with the default).
2. **Write a prompt** — e.g. `red cubes`, `the blue one on the left`.
3. **Pick a detection type** — `2D bounding boxes`, `Segmentation masks`, or `Points`.
4. **Pick a model** — dropdown includes Gemini ER, Gemini 3, Claude Sonnet/Opus, GPT-5.
5. **(Optional) toggle knobs**:
   - *Optimised prompts* — swaps in per-provider tuned prompts.
   - *Two-pass* — does an overview pass then zooms into each region; better on cluttered scenes.
   - *Thinking* — enables extended reasoning (Anthropic) or Gemini thinking mode.
   - *Temperature* slider.
6. **Send** — runs the loop on the selected model and picks up whatever it detected.

**Benchmark all models at once:** click **Run All Models**. It runs every configured model across `2D bounding boxes` and `Points` on the current scene, logs each result, and exports per-run screenshots you can compare side-by-side.

## Security

`vite.config.ts` inlines your API keys into the client bundle at build time, so `dist/` contains them in plaintext — **don't host the build publicly**. For real deployments, proxy provider calls through your own backend.

## License

MIT — see [LICENSE](LICENSE).
