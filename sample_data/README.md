# Sample data

- **`sample_scene.png`** — a synthetic scene (384×384) of colored shapes produced by
  `mmqa.data.synth_scene`. It embeds the scene spec (shapes, colors, boxes) in the PNG metadata,
  so the offline `SceneStubVQA` can answer questions about it without a real model. Try it:

  ```bash
  # answer a question about the image (uses the real ViLT model; downloads on first use)
  mmqa ask --image sample_data/sample_scene.png --question "how many shapes are there?"

  # offline (no model): answer about a built-in seed scene with the scene stub
  mmqa ask-scene --question "what color is the circle?" --scene 0 --fast

  # run the agent on all seed scenes
  mmqa demo-agent --fast
  ```

> `--fast` uses the SceneStubVQA (no download). Drop it to use `dandelin/vilt-b32-finetuned-vqa`.
> On a real photo, install the ML extra (`pip install -e .[ml]`) so the ViLT model runs.
