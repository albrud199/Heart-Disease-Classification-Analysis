# Record Demo GIF for Your Resume/Portfolio

Demo shows the Gradio app in action. Tools to record: **ScreenToGif** (Windows) or **ffmpeg**.

## Option 1: ScreenToGif (Easiest for Windows)

1. Download **ScreenToGif**: https://www.screentogif.com/
2. Run the app and click "Recorder"
3. Start the Gradio app locally:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:7860 in browser
5. In ScreenToGif, position the window and press Start
6. Enter a sample input (e.g., `63,1,3,145,233,1,0,150,0,2.3,0,0,1`)
7. Press Submit and wait for result (~1-2 seconds)
8. Click Stop in ScreenToGif
9. Save as GIF (max 5 seconds recommended)

## Option 2: FFmpeg (CLI)

**Windows PowerShell:**
```bash
# Record 8 seconds of screen (adjust monitor/resolution as needed)
ffmpeg -f gdigrab -framerate 30 -i desktop -t 8 demo.mp4

# Convert to GIF
ffmpeg -i demo.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" demo.gif
```

**Linux/Mac:**
```bash
# Record screen
ffmpeg -f x11grab -framerate 30 -i :0.0 -t 8 demo.mp4

# Convert to GIF
ffmpeg -i demo.mp4 -vf "fps=10,scale=640:-1" demo.gif
```

## What to Show in Demo (30 seconds)

1. **App loads** (3 sec) — show Gradio interface
2. **Enter input** (3 sec) — paste example features
3. **Submit** (2 sec) — click "Submit"
4. **Result appears** (2 sec) — show prediction + probability
5. **Explain** (1 sec) — mention "Model deployed on Hugging Face Spaces"

## Upload GIF to GitHub

1. Save GIF as `demo.gif` in repo root
2. Add to README.md:
   ```markdown
   ## Demo

   ![Heart Disease Prediction Demo](demo.gif)
   ```

## Resume Bullet with Demo

*Built and deployed a heart disease prediction app using Gradio. The model achieves 95% accuracy with explanations via SHAP. Live demo: [link to HF Space]. GitHub: [link to repo].*

Or simply add the GIF to your GitHub README and recruiters will see it instantly.
