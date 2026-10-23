import { useCallback, useEffect, useRef, useState } from "react";
import { Tensor } from "../lib/autograd";
import { MLP, spirals, type Dataset } from "../lib/mlp";

const CLASS_RGB: [number, number, number][] = [
  [52, 110, 190],
  [214, 120, 40],
  [40, 150, 120],
];
const EXTENT = 1.2;
const GRID = 90;
const MAX_STEPS = 3000;

/** Trains a ReLU MLP on the spirals with full-batch gradient descent, entirely in the browser. */
export function SpiralTrainer() {
  const data = useRef<Dataset>(spirals(100, 3, 1));
  const model = useRef<MLP | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const grid = useRef<Tensor | null>(null);
  const [width, setWidth] = useState(64);
  const [lr, setLr] = useState(0.5);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState({ step: 0, loss: NaN, acc: NaN });
  const [history, setHistory] = useState<number[]>([]);
  const lrRef = useRef(lr);
  const stepRef = useRef(0);
  lrRef.current = lr;

  const draw = useCallback(() => {
    const el = canvas.current;
    const m = model.current;
    if (!el || !m) return;
    const ctx = el.getContext("2d")!;
    const size = el.width;
    if (!grid.current) {
      const pts: number[][] = [];
      for (let i = 0; i < GRID; i++)
        for (let j = 0; j < GRID; j++)
          pts.push([-EXTENT + ((j + 0.5) * 2 * EXTENT) / GRID, EXTENT - ((i + 0.5) * 2 * EXTENT) / GRID]);
      grid.current = Tensor.from(pts);
    }
    const pred = m.predict(grid.current);
    const img = ctx.createImageData(GRID, GRID);
    pred.forEach((k, idx) => {
      const [r, g, b] = CLASS_RGB[k];
      img.data.set([r, g, b, 70], idx * 4);
    });
    const off = new OffscreenCanvas(GRID, GRID);
    off.getContext("2d")!.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, size, size);

    const { X, labels } = data.current;
    for (let i = 0; i < labels.length; i++) {
      const px = ((X.data[2 * i] + EXTENT) / (2 * EXTENT)) * size;
      const py = ((EXTENT - X.data[2 * i + 1]) / (2 * EXTENT)) * size;
      const [r, g, b] = CLASS_RGB[labels[i]];
      ctx.beginPath();
      ctx.arc(px, py, size / 110, 0, 2 * Math.PI);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fill();
      ctx.lineWidth = size / 400;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.stroke();
    }
  }, []);

  const reset = useCallback(
    (w: number) => {
      model.current = new MLP([2, w, w, 3], 7);
      stepRef.current = 0;
      setRunning(false);
      setHistory([]);
      setStats({ step: 0, loss: NaN, acc: model.current.accuracy(data.current) });
      draw();
    },
    [draw],
  );

  useEffect(() => {
    const el = canvas.current!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.width = el.height = Math.round(340 * dpr);
    reset(width);
  }, []);

  useEffect(() => {
    if (!running) return;
    const m = model.current!;
    let timer = 0;
    let lastDraw = 0;
    let loss = NaN;
    const losses: number[] = [];
    const publish = () => {
      draw();
      setStats({ step: stepRef.current, loss, acc: m.accuracy(data.current) });
      const fresh = losses.splice(0); // outside the updater, which React may call twice
      setHistory((h) => [...h, ...fresh]);
    };
    // Train in ~14 ms slices and yield between them. setTimeout rather than requestAnimationFrame, so training
    // speed doesn't depend on the display's frame rate.
    const slice = () => {
      const start = performance.now();
      while (performance.now() - start < 14 && stepRef.current < MAX_STEPS) {
        loss = m.step(data.current, lrRef.current);
        stepRef.current++;
        if (stepRef.current % 10 === 0) losses.push(loss);
      }
      if (start - lastDraw > 200 || stepRef.current >= MAX_STEPS) {
        lastDraw = start;
        publish();
      }
      if (stepRef.current >= MAX_STEPS) {
        setRunning(false);
        return;
      }
      timer = window.setTimeout(slice, 1);
    };
    timer = window.setTimeout(slice, 0);
    return () => {
      window.clearTimeout(timer);
      // on pause, show the exact state; after a reset the model has been replaced and there is nothing to show
      if (model.current === m && !Number.isNaN(loss)) publish();
    };
  }, [running, draw]);

  return (
    <div className="demo">
      <p className="demo-title">Interactive · training with the mini autograd</p>
      <p className="demo-lede">
        A 2 → w → w → 3 ReLU network with He initialization, trained by full-batch gradient descent on mean
        cross-entropy. Every gradient comes from the same <code>backward()</code> that passed the checks above. Shaded
        regions show the class the network predicts.
      </p>
      <div className="trainer">
        <canvas ref={canvas} aria-label="Spiral data and the network's decision regions" />
        <div>
          <div className="demo-row">
            <button onClick={() => setRunning((r) => !r)} disabled={stats.step >= MAX_STEPS}>
              {running ? "Pause" : stats.step === 0 ? "Train" : "Resume"}
            </button>
            <button className="secondary" onClick={() => reset(width)}>
              Reset
            </button>
          </div>
          <div className="demo-row">
            <label>
              Learning rate
              <input type="range" min={0.05} max={1.5} step={0.05} value={lr} onChange={(e) => setLr(Number(e.target.value))} />
              <span className="num">{lr.toFixed(2)}</span>
            </label>
          </div>
          <div className="demo-row">
            <label>
              Hidden width w
              <select
                value={width}
                onChange={(e) => {
                  const w = Number(e.target.value);
                  setWidth(w);
                  reset(w);
                }}
              >
                {[4, 8, 16, 32, 64].map((w) => (
                  <option key={w} value={w}>
                    {w}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <dl className="stats">
            <dt>Step</dt>
            <dd className="num">
              {stats.step} / {MAX_STEPS}
            </dd>
            <dt>Loss</dt>
            <dd className="num">{Number.isNaN(stats.loss) ? "–" : stats.loss.toFixed(4)}</dd>
            <dt>Accuracy</dt>
            <dd className="num">{Number.isNaN(stats.acc) ? "–" : `${(100 * stats.acc).toFixed(1)}%`}</dd>
          </dl>
          <LossChart losses={history} />
        </div>
      </div>
      <p className="note">
        Things to try: width 4 tops out around 75%, with too few units to carve out three spirals, while width 8 already
        reaches about 99%. At width 8, learning rates of 1.2 or more stall near chance, but width 64 still trains at 1.5.
      </p>
    </div>
  );
}

function LossChart({ losses }: { losses: number[] }) {
  const w = 260;
  const h = 110;
  const pad = 22;
  if (losses.length < 2) {
    return (
      <svg className="loss-chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Loss curve">
        <line className="axis" x1={pad} y1={h - pad} x2={w - 4} y2={h - pad} />
        <text x={pad} y={h - 6}>
          loss (log scale) appears here
        </text>
      </svg>
    );
  }
  const logs = losses.map((l) => Math.log10(Math.max(l, 1e-6)));
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  const span = Math.max(hi - lo, 1e-9);
  const pts = logs.map((v, i) => {
    const x = pad + (i / (MAX_STEPS / 10 - 1)) * (w - pad - 4);
    const y = 6 + ((hi - v) / span) * (h - pad - 10);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg className="loss-chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Loss curve on a log scale">
      <line className="axis" x1={pad} y1={h - pad} x2={w - 4} y2={h - pad} />
      <line className="axis" x1={pad} y1={4} x2={pad} y2={h - pad} />
      <path d={`M${pts.join(" L")}`} />
      <text x={pad + 2} y={h - 6}>
        step → {MAX_STEPS}
      </text>
      <text x={2} y={12}>
        {(10 ** hi).toFixed(2)}
      </text>
      <text x={2} y={h - pad}>
        {(10 ** lo).toFixed(3)}
      </text>
    </svg>
  );
}
