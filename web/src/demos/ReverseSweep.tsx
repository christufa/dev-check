import { useMemo, useState } from "react";

/**
 * f(x1, x2, x3) = tanh(x1·x2) + x2·x3²  as a graph. Step through the reverse sweep to watch gradients flow
 * backwards and accumulate at x2 and x3, which fan out; then compare with the three forward-mode passes.
 */

type Id = "x1" | "x2" | "x3" | "a" | "b" | "c" | "d" | "f";

interface NodeSpec {
  id: Id;
  label: string;
  x: number;
  y: number;
}

const NODES: NodeSpec[] = [
  { id: "x1", label: "x₁", x: 80, y: 55 },
  { id: "x2", label: "x₂", x: 80, y: 160 },
  { id: "x3", label: "x₃", x: 80, y: 265 },
  { id: "a", label: "a = x₁·x₂", x: 245, y: 95 },
  { id: "c", label: "c = x₃·x₃", x: 245, y: 265 },
  { id: "b", label: "b = tanh a", x: 420, y: 95 },
  { id: "d", label: "d = x₂·c", x: 420, y: 215 },
  { id: "f", label: "f = b + d", x: 600, y: 155 },
];

/** One edge per (parent → child) use; x3 feeds c twice. `bend` separates the duplicate edges. */
const EDGES: { from: Id; to: Id; bend?: number }[] = [
  { from: "x1", to: "a" },
  { from: "x2", to: "a" },
  { from: "x2", to: "d" },
  { from: "x3", to: "c", bend: -22 },
  { from: "x3", to: "c", bend: 22 },
  { from: "c", to: "d" },
  { from: "a", to: "b" },
  { from: "b", to: "f" },
  { from: "d", to: "f" },
];

type Vals = Record<Id, number>;

function forward(x1: number, x2: number, x3: number): Vals {
  const a = x1 * x2;
  const b = Math.tanh(a);
  const c = x3 * x3;
  const d = x2 * c;
  return { x1, x2, x3, a, b, c, d, f: b + d };
}

/**
 * Reverse topological order of the non-leaf nodes, and the local partial ∂child/∂parent for each edge it fires.
 * Processing node t pushes g_t · ∂t/∂p onto each parent p.
 */
function sweep(v: Vals): { node: Id; pushes: { to: Id; edge: number; amount: (g: number) => number }[] }[] {
  return [
    {
      node: "f",
      pushes: [
        { to: "b", edge: 7, amount: (g) => g },
        { to: "d", edge: 8, amount: (g) => g },
      ],
    },
    {
      node: "d",
      pushes: [
        { to: "x2", edge: 2, amount: (g) => g * v.c },
        { to: "c", edge: 5, amount: (g) => g * v.x2 },
      ],
    },
    {
      node: "c",
      pushes: [
        { to: "x3", edge: 3, amount: (g) => g * v.x3 },
        { to: "x3", edge: 4, amount: (g) => g * v.x3 },
      ],
    },
    { node: "b", pushes: [{ to: "a", edge: 6, amount: (g) => g * (1 - v.b * v.b) }] },
    {
      node: "a",
      pushes: [
        { to: "x1", edge: 0, amount: (g) => g * v.x2 },
        { to: "x2", edge: 1, amount: (g) => g * v.x1 },
      ],
    },
  ];
}

/** Forward mode with dual numbers: one pass per input, seeding that input's tangent with 1. */
function forwardModePass(x: [number, number, number], seed: 0 | 1 | 2): number {
  const t = [0, 0, 0];
  t[seed] = 1;
  const a = x[0] * x[1];
  const ta = t[0] * x[1] + x[0] * t[1];
  const tb = (1 - Math.tanh(a) ** 2) * ta;
  const c = x[2] * x[2];
  const tc = 2 * x[2] * t[2];
  const td = t[1] * c + x[1] * tc;
  return tb + td;
}

const fmt = (n: number) => (Math.abs(n) < 5e-4 ? "0" : n.toFixed(3));

export function ReverseSweep() {
  const [x, setX] = useState<[number, number, number]>([0.5, -1.2, 2.0]);
  const steps = 5;
  const [step, setStep] = useState(steps);
  const v = useMemo(() => forward(...x), [x]);
  const plan = useMemo(() => sweep(v), [v]);

  // gradients after `step` nodes have been processed
  const grads: Partial<Record<Id, number>> = { f: 1 };
  const contributions: Partial<Record<Id, number>> = {};
  let firing: Set<number> = new Set();
  let firingNode: Id | null = null;
  plan.slice(0, step).forEach((s, k) => {
    for (const p of s.pushes) {
      grads[p.to] = (grads[p.to] ?? 0) + p.amount(grads[s.node]!);
      contributions[p.to] = (contributions[p.to] ?? 0) + 1;
    }
    if (k === step - 1) {
      firing = new Set(s.pushes.map((p) => p.edge));
      firingNode = s.node;
    }
  });
  const done = new Set<Id>(["f", ...plan.slice(0, step).map((s) => s.node)]);

  const pos = Object.fromEntries(NODES.map((n) => [n.id, n])) as Record<Id, NodeSpec>;
  const W = 150;
  const H = 54;

  const fwd = [0, 1, 2].map((i) => forwardModePass(x, i as 0 | 1 | 2));

  return (
    <div className="demo">
      <p className="demo-title">Interactive · reverse sweep on a graph</p>
      <p className="demo-lede">
        Each box shows its forward value <span className="num">v</span> and, once the sweep reaches it, its gradient{" "}
        <span className="num" style={{ color: "var(--rev)" }}>
          g = ∂f/∂v
        </span>
        . Watch <span className="num">x₂</span> and <span className="num">x₃</span>: they feed two edges each, so their
        gradients arrive in two pieces and are summed.
      </p>
      <div className="demo-row">
        {(["x₁", "x₂", "x₃"] as const).map((name, i) => (
          <label key={name}>
            {name}
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={x[i]}
              onChange={(e) => {
                const next = [...x] as [number, number, number];
                next[i] = Number(e.target.value);
                setX(next);
              }}
            />
            <span className="num">{x[i].toFixed(1)}</span>
          </label>
        ))}
      </div>
      <svg viewBox="0 0 690 320" role="img" aria-label="Computation graph of f with values and gradients">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-faint)" />
          </marker>
        </defs>
        {EDGES.map((e, i) => {
          const p = pos[e.from];
          const q = pos[e.to];
          const x1 = p.x + W / 2;
          const x2 = q.x - W / 2;
          const mx = (x1 + x2) / 2;
          const bend = e.bend ?? 0;
          return (
            <path
              key={i}
              className={`graph-edge${firing.has(i) ? " firing" : ""}`}
              d={`M${x1},${p.y} C${mx},${p.y + bend} ${mx},${q.y + bend} ${x2},${q.y}`}
              markerEnd="url(#arrow)"
            />
          );
        })}
        {NODES.map((n) => {
          const g = grads[n.id];
          const cls = n.id === firingNode ? "firing" : done.has(n.id) || g !== undefined ? "done" : "";
          return (
            <g key={n.id} className={`graph-node ${cls}`} transform={`translate(${n.x - W / 2},${n.y - H / 2})`}>
              <rect width={W} height={H} rx={8} />
              <text className="label" x={10} y={18}>
                {n.label}
              </text>
              <text className="val" x={10} y={33}>
                v = {fmt(v[n.id])}
              </text>
              <text className="grad" x={10} y={47}>
                {g === undefined ? "g = ·" : `g = ${fmt(g)}`}
                {(contributions[n.id] ?? 0) > 1 ? `  (${contributions[n.id]} parts)` : ""}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="demo-row">
        <button className="secondary" onClick={() => setStep(0)} disabled={step === 0}>
          Reset
        </button>
        <button className="secondary" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>
          Back
        </button>
        <button onClick={() => setStep(Math.min(steps, step + 1))} disabled={step === steps}>
          Process next node
        </button>
        <span>
          {step === 0
            ? "Seeded g_f = 1. Nothing processed yet."
            : `Processed ${plan
                .slice(0, step)
                .map((s) => s.node)
                .join(", ")}${step === steps ? ". The sweep is complete." : ""}`}
        </span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Input</th>
            <th>Reverse mode (1 sweep)</th>
            <th>Forward mode</th>
          </tr>
        </thead>
        <tbody>
          {(["x1", "x2", "x3"] as const).map((id, i) => (
            <tr key={id}>
              <td className="num">∂f/∂{["x₁", "x₂", "x₃"][i]}</td>
              <td className="num">{step === steps ? fmt(grads[id]!) : "pending"}</td>
              <td className="num">
                {fmt(fwd[i])} <span style={{ color: "var(--ink-faint)" }}>(pass {i + 1}, seed e{["₁", "₂", "₃"][i]})</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        Forward mode needed three passes, one per input. Reverse mode got all three partials from a single sweep. With P
        parameters, that's P passes against one.
      </p>
    </div>
  );
}
