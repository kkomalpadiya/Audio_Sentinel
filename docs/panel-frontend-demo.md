# Local frontend presentation demo

The frontend displays real results from `AudioFeatureService.prepare`: generated
audio is prepared, windowed, saved, converted to Log-Mel features, and reloaded
through `load_log_mel`. The browser draws those actual float32 values. This demo
does not run YAMNet or infer acoustic labels, threat scores, or alerts.

## Start

Open PowerShell and run:

```powershell
Set-Location 'C:\Users\kkoma\OneDrive\Desktop\Project_1'
& 'C:\Users\kkoma\miniconda3\python.exe' -m uvicorn audio_sentinel.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/demo in your browser. Keep the terminal running. If the
previous demonstration server is still running, stop it with Ctrl+C first so
the newly added routes are loaded. Stop this server with Ctrl+C after the demo.
All dependencies already exist in the verified local interpreter. No Node build,
internet connection, model download, or additional Python installation is needed.

## Five-minute speaking sequence

1. **Person A:** “This is our local feature explorer. It calls the implemented
   Python pipeline and displays its actual results.” The first frequency sweep
   runs automatically. Initial processing can take longer while libraries load.
2. **Person B:** Point out **48 → 16 kHz**, **stereo → mono**, **five windows**,
   and **64 Mel bands**. Play the original and prepared audio at a low volume.
   “The waveform shows prepared amplitude over time. The spectrogram shows
   frequency-band energy over time.”
3. **Person B:** Choose **Steady tone**, then **Generate spectrogram**. Point out
   the horizontal band. Choose **Frequency sweep**, generate again, and point
   out the rising band. These are known synthetic signals, not detected events.
4. **Person A:** Select the **10 s window**. “Our source lasts 1.6 seconds, so
   the remaining 8.4 seconds are zero-padded. The dashed line marks where real
   audio ends.” Hover on the plot to inspect the time, band, frame, and dB value.
5. **Person A:** Point out the integrity checks. “We save and verify the feature
   bundles, preserve the source, and confirm reuse on a repeat call.” Use
   **Save image** to keep the current plot for the presentation.
6. **Person A:** “Preparation and feature extraction are complete. Model loading
   and inference come next. YAMNet will use its own compatible frontend.”

## Interpretation and implementation

- The vertical axis is Mel **band number**, not linearly spaced hertz.
- The horizontal axis shows time relative to the selected window. Pixel columns
  are centered at the actual FFT frame times and displayed at one hop width.
  Small dark strips at the ends have no analyzed frame center.
- The colorbar uses power dB relative to the configured reference power 1.0;
  these are neither calibrated sound-pressure levels nor class probabilities.
  Color limits are shared across windows within a run, but may change between
  generated samples. Feature clipping follows the existing per-window recipe.
- The waveform is a 640-bin min/max amplitude envelope, retaining short peaks.
  Its highlighted region is the selected window's real source interval.
- Browser audio controls play actual generated/prepared PCM16 WAV data. Playback
  is user-triggered; only one player runs at a time. Nothing is recorded.
- Requests accept only `tone`, `sweep`, or `pulses`. The pulses use a fixed random
  seed. This demo has no file-upload or arbitrary-file access endpoint.
- Each request uses its own temporary directory. It processes the same source
  twice to verify reuse, reloads verified features, then removes the temporary
  files. The measured duration includes both calls and verification; it is not
  a model inference benchmark. A server lock allows only one demo run at a time.
- The generated WAV and prepared WAV are returned to the browser for playback.
  Feature arrays retain their float32 values when serialized. There is no
  persistent demo database. The downloaded PNG contains the plot and axes.
- Files: `src/audio_sentinel/demo.py` (demo adapter), `web/index.html`,
  `web/style.css`, and `web/app.js` (under `src/audio_sentinel`). `main.py` mounts
  the routes. Existing health and project-status endpoints remain available.

## Checks

```powershell
python -m pytest tests/test_demo.py -q
python -m pytest -q
```

The demo tests compare API arrays with arrays actually reloaded from saved feature
bundles, decode the returned WAV files, check generated spectral behavior and
padding, and exercise input rejection, concurrent-run rejection, and cleanup on
failure. The frontend has no external fonts, scripts, or stylesheets.

The optional, feature-detected WebMCP action uses the same form action. A supported
WebMCP browser context was not available for its contract verification. Browser
visual/interaction QA was not performed; verification covered the HTTP routes,
actual numerical/audio output, Python tests, and JavaScript syntax.
