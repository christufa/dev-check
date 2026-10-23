import { useState } from "react";
import { TOLERANCE, mlpCheck, opChecks, type CheckResult } from "../lib/checks";

/** Runs the lesson's gradient checks in the browser, using the TypeScript port of the engine. */
export function GradCheck() {
  const [results, setResults] = useState<CheckResult[] | null>(null);
  const [zeroBiases, setZeroBiases] = useState(true);
  const [mlp, setMlp] = useState<{ zero: boolean; error: number } | null>(null);

  return (
    <div className="demo">
      <p className="demo-title">Interactive · gradient checks</p>
      <p className="demo-lede">
        Each row builds a scalar from random 4×3 inputs, runs <code>backward()</code>, and compares the result with
        central differences (h = 10⁻⁶) for every input entry. The engine is a TypeScript port of the NumPy one in the
        listing above.
      </p>
      <div className="demo-row">
        <button onClick={() => setResults(opChecks(Math.floor(Math.random() * 1e9)))}>
          {results ? "Re-run with new random inputs" : "Run the op checks"}
        </button>
      </div>
      {results && <ResultTable rows={results} />}

      <div className="demo-row" style={{ marginTop: "1.2rem" }}>
        <label>
          <input type="checkbox" checked={zeroBiases} onChange={(e) => setZeroBiases(e.target.checked)} />
          Zero biases
        </label>
        <button onClick={() => setMlp({ zero: zeroBiases, error: mlpCheck(zeroBiases) })}>Check the whole MLP</button>
      </div>
      {mlp && (
        <>
          <ResultTable rows={[{ name: `2→8→8→3 MLP, ${mlp.zero ? "zero" : "random"} biases`, error: mlp.error }]} />
          {mlp.zero && mlp.error > TOLERANCE && (
            <p className="note">
              This failure isn't a bug. The first spiral point is the origin, so with zero biases every first-layer
              pre-activation there is exactly 0, which is the ReLU kink. The engine returns the subgradient 𝟙[z &gt; 0] = 0,
              while the central difference averages the two one-sided slopes and gets ½. Untick "Zero biases" and the
              check passes.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function ResultTable({ rows }: { rows: CheckResult[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Check</th>
          <th>Max relative error</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.name}>
            <td>{r.name}</td>
            <td className="num">{r.error.toExponential(2)}</td>
            <td className={r.error < TOLERANCE ? "ok" : "fail"}>{r.error < TOLERANCE ? "ok" : "fail"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
