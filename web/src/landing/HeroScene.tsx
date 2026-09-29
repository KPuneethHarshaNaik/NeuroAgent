import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { heroWave, sampleWave } from "@/landing/wave";

const SAMPLES = 900;

/** How much of the visible half-height the trace occupies. */
const AMPLITUDE = 0.82;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

/**
 * The one big motion moment on the page: a drawn EEG trace in the signal accent, on near-black,
 * that reacts to the cursor (a probe that lifts the local amplitude) and to scroll (the trace
 * compresses and slides as the hero leaves).
 *
 * Deliberately careful about frames:
 *  - it paints one frame synchronously on mount, so it never appears blank if rAF is throttled;
 *  - the loop stops while the hero is off-screen or the tab is hidden;
 *  - `prefers-reduced-motion` renders a single static frame and never starts a loop;
 *  - with no WebGL context it falls back to the same waveform drawn as SVG.
 */
export function HeroScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || reduced) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    } catch {
      setFailed(true);
      return;
    }
    if (!renderer.getContext()) {
      setFailed(true);
      return;
    }

    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 2;

    const trace = (opacity: number, offset: number) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(SAMPLES * 3), 3));
      const material = new THREE.LineBasicMaterial({
        color: 0xe4293f,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const line = new THREE.Line(geometry, material);
      line.position.y = offset;
      scene.add(line);
      return line;
    };

    // linewidth is unreliable across drivers, so weight comes from stacked copies of the same
    // trace offset by a constant distance: they read as one glowing stroke.
    const passes = [
      { line: trace(0.95, 0), offset: 0 },
      { line: trace(0.34, 0.005), offset: 0.005 },
      { line: trace(0.34, -0.005), offset: -0.005 },
      { line: trace(0.12, 0.012), offset: 0.012 },
      { line: trace(0.12, -0.012), offset: -0.012 },
    ];

    // Baseline + a hairline grid, so the trace sits on an instrument rather than a gradient.
    // Rebuilt on resize because its width has to track the visible camera width.
    const gridGeometry = new THREE.BufferGeometry();
    const grid = new THREE.LineSegments(
      gridGeometry,
      new THREE.LineBasicMaterial({ color: 0x1e262c, transparent: true, opacity: 0.9, depthWrite: false }),
    );
    scene.add(grid);

    const buildGrid = () => {
      const half = camera.right;
      gridGeometry.setFromPoints(
        [-0.8, -0.4, 0, 0.4, 0.8].flatMap((y) => [new THREE.Vector3(-half, y, 0), new THREE.Vector3(half, y, 0)]),
      );
    };

    // The probe: a vertical hairline that follows the cursor across the trace.
    const probeGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 1, 0)]);
    const probe = new THREE.Line(
      probeGeometry,
      new THREE.LineBasicMaterial({ color: 0xe4293f, transparent: true, opacity: 0.18, depthWrite: false }),
    );
    scene.add(probe);

    let cursorX = 0.05;
    let cursorTargetX = 0.05;
    let cursorY = 0;
    let cursorTargetY = 0;
    let gain = 1;
    let gainTarget = 1;
    let scroll = 0;
    let scrollTarget = 0;
    let visible = true;
    let frame = 0;
    let last = 0;
    let time = 0;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      const aspect = width / Math.max(height, 1);
      camera.left = -aspect;
      camera.right = aspect;
      camera.top = 1;
      camera.bottom = -1;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(width, height, false);
      buildGrid();
      draw(0);
    };

    const draw = (dt: number) => {
      time += dt;
      cursorX += (cursorTargetX - cursorX) * Math.min(1, dt * 4);
      cursorY += (cursorTargetY - cursorY) * Math.min(1, dt * 4);
      gain += (gainTarget - gain) * Math.min(1, dt * 3);
      scroll += (scrollTarget - scroll) * Math.min(1, dt * 3);

      // The waveform is defined over -1..1; it is stretched across the visible width so a narrow
      // viewport shows the whole signal instead of a clipped middle slice.
      const half = camera.right;
      for (const { line, offset } of passes) {
        const attribute = line.geometry.getAttribute("position") as THREE.BufferAttribute;
        const array = attribute.array as Float32Array;
        for (let i = 0; i < SAMPLES; i += 1) {
          const u = (i / (SAMPLES - 1)) * 2 - 1;
          const wave = heroWave(u + scroll * 0.18, time, cursorX / half, gain) * AMPLITUDE;
          array[i * 3] = u * half;
          array[i * 3 + 1] = clamp(wave - scroll * 0.12 + offset - cursorY * 0.02, -0.97, 0.97);
          array[i * 3 + 2] = 0;
        }
        attribute.needsUpdate = true;
      }

      probe.position.x = cursorX;
      probe.material.opacity = 0.06 + 0.2 * Math.abs(cursorX);
      renderer.render(scene, camera);
    };

    const tick = (now: number) => {
      if (!visible) return;
      frame = requestAnimationFrame(tick);
      const dt = last ? Math.min((now - last) / 1000, 0.05) : 1 / 60;
      last = now;
      draw(dt);
    };

    const start = () => {
      if (!visible || frame) return;
      last = 0;
      frame = requestAnimationFrame(tick);
    };
    const stop = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const aspect = camera.right;
      cursorTargetX = ((event.clientX - rect.left) / rect.width - 0.5) * 2 * aspect;
      cursorTargetY = -(((event.clientY - rect.top) / rect.height - 0.5) * 2);
      gainTarget = 1 + Math.min(Math.abs(cursorTargetX), 1) * 0.15;
    };
    const onPointerLeave = () => {
      cursorTargetX = 0.05;
      cursorTargetY = 0;
      gainTarget = 1;
    };
    const onScroll = () => {
      const hero = canvas.parentElement?.parentElement;
      const span = hero ? Math.max(hero.offsetHeight, 1) : window.innerHeight;
      scrollTarget = Math.min(window.scrollY / span, 1);
    };
    const onVisibility = () => {
      visible = document.visibilityState === "visible";
      if (visible) start();
      else stop();
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting && document.visibilityState === "visible";
        if (visible) start();
        else stop();
      },
      { threshold: 0.05 },
    );
    observer.observe(canvas);

    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerleave", onPointerLeave);
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    onScroll();
    resize();
    start();

    return () => {
      stop();
      observer.disconnect();
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerleave", onPointerLeave);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("visibilitychange", onVisibility);
      for (const { line } of passes) {
        line.geometry.dispose();
        (line.material as THREE.Material).dispose();
      }
      gridGeometry.dispose();
      probeGeometry.dispose();
      renderer.dispose();
    };
  }, [reduced]);

  if (reduced || failed) {
    return <StaticTrace />;
  }

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      aria-hidden="true"
      data-testid="hero-canvas"
    />
  );
}

/** Same waveform, drawn once: used for reduced motion and for machines without WebGL. */
function StaticTrace() {
  const points = sampleWave(320, { t: 0.6, cursorX: 0.12, gain: 1 });
  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${((point.x + 1) / 2 * 1000).toFixed(1)} ${(50 - point.y * 34).toFixed(1)}`)
    .join(" ");
  return (
    <svg
      className="absolute inset-0 h-full w-full"
      viewBox="0 0 1000 100"
      preserveAspectRatio="none"
      aria-hidden="true"
      data-testid="hero-svg-fallback"
    >
      {[20, 40, 60, 80].map((y) => (
        <line key={y} x1="0" y1={y} x2="1000" y2={y} stroke="#1E262C" strokeWidth="1" />
      ))}
      <path d={path} fill="none" stroke="#E4293F" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const listener = () => setReduced(query.matches);
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, []);
  return reduced;
}
