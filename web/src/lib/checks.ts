import { Tensor, crossEntropy, gradCheck, makeRng } from "./autograd";
import { MLP, spirals } from "./mlp";

export interface CheckResult {
  name: string;
  error: number;
}

/** Relative errors below this mean the gradient is right (see the gradient-checking section). */
export const TOLERANCE = 1e-6;

/** The per-op gradient checks from the lesson, each a scalar built from random inputs. */
export function opChecks(seed = 0): CheckResult[] {
  const rng = makeRng(seed);
  const r = (rows: number, cols: number) => Tensor.randn(rows, cols, 1, rng);
  const A = r(4, 3);
  const B = r(4, 3);
  const R = r(4, 3);
  const W = r(3, 5);
  const C = r(4, 5);
  const bRow = r(1, 3);
  const col = r(4, 1);

  const cases: [string, (...xs: Tensor[]) => Tensor, Tensor[]][] = [
    ["add, broadcast 1×3", (a, b) => a.add(b).tanh().mul(R).sum(), [A, bRow]],
    ["mul, broadcast 4×1", (a, b) => a.mul(b).tanh().mul(R).sum(), [A, col]],
    ["div and pow", (a, b) => a.div(b.pow(2).add(1)).mul(R).sum(), [A, B]],
    ["matmul", (a, w) => a.matmul(w).tanh().mul(C).sum(), [A, W]],
    ["exp and log", (a) => a.exp().add(1).log().mul(R).mean(), [A]],
    ["relu", (a, b) => a.relu().mul(b).sum(), [A, B]],
    ["sum over an axis", (a) => a.sum(0).pow(2).sum(), [A]],
    ["mean over an axis", (a) => a.mean(1).mul(a).tanh().sum(), [A]],
    ["log-softmax", (a) => a.logSoftmax().mul(R).sum(), [A]],
    ["fan-out (input used 3 times)", (a) => a.mul(a).tanh().mul(a).mul(R).sum(), [A]],
  ];
  return cases.map(([name, build, inputs]) => ({ name, error: gradCheck(build, ...inputs) }));
}

/**
 * End-to-end check of a small MLP with cross-entropy on 20 spiral points.
 * The first spiral point is the origin, so with zero biases every first-layer ReLU there sits exactly on its kink.
 */
export function mlpCheck(zeroBiases: boolean): number {
  const data = spirals(100, 3, 1);
  const n = 20;
  const X = new Tensor(data.X.data.slice(0, n * 2), n, 2);
  const Y = new Tensor(data.Y.data.slice(0, n * 3), n, 3);
  const model = new MLP([2, 8, 8, 3], 2);
  if (!zeroBiases) {
    const rng = makeRng(3);
    for (const b of model.params.filter((_, i) => i % 2 === 1)) b.data.forEach((_, i) => (b.data[i] = 0.1 * (rng() - 0.5)));
  }
  return gradCheck((...ps) => crossEntropy(model.forward(X, ps), Y), ...model.params);
}
