/**
 * The hero trace, as a pure function so the WebGL scene and the no-WebGL SVG fallback draw the
 * same signal. Nothing here claims to be a recording: it is a drawn waveform in the product's
 * accent, and it is labelled as such on the page.
 */

/** Two narrow transients riding on three rhythms -- the shape of an event-locked average, drawn. */
export function heroWave(x: number, t: number, cursorX: number, gain: number): number {
  const rhythm =
    0.17 * Math.sin(x * 6.1 + t * 1.5) + 0.09 * Math.sin(x * 14.3 - t * 2.2) + 0.045 * Math.sin(x * 31.0 + t * 3.3);
  const transient = -0.62 * bump(x, -0.18 + 0.06 * Math.sin(t * 0.35), 0.028) + 0.5 * bump(x, 0.3, 0.035);
  const probe = 1 + 1.5 * bump(x, cursorX, 0.075);
  return (rhythm + transient * 0.55) * gain * probe;
}

function bump(x: number, center: number, width: number): number {
  const d = (x - center) / width;
  return Math.exp(-0.5 * d * d);
}

export type WavePoint = { x: number; y: number };

/** Sample the trace across a normalised -1..1 window, for the SVG fallback. */
export function sampleWave(count: number, options: { t?: number; cursorX?: number; gain?: number } = {}): WavePoint[] {
  const { t = 0.6, cursorX = 0.1, gain = 1 } = options;
  const points: WavePoint[] = [];
  for (let i = 0; i < count; i += 1) {
    const x = (i / (count - 1)) * 2 - 1;
    points.push({ x, y: heroWave(x, t, cursorX, gain) });
  }
  return points;
}
