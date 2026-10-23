import { describe, expect, it } from "vitest";
import { Tensor, crossEntropy, makeRng } from "./autograd";
import { TOLERANCE, mlpCheck, opChecks } from "./checks";
import { MLP, spirals } from "./mlp";

describe("autograd", () => {
  it("accumulates fan-out: y = x·x + x has dy/dx = 2x + 1", () => {
    const x = Tensor.scalar(3);
    x.mul(x).add(x).backward();
    expect(x.grad![0]).toBe(7);
  });

  it.each(opChecks())("matches central differences: $name", ({ error }) => {
    expect(error).toBeLessThan(TOLERANCE);
  });

  it("gives (S − Y)/N at the logits for softmax + cross-entropy", () => {
    const rng = makeRng(5);
    const N = 6;
    const K = 4;
    const logits = Tensor.randn(N, K, 1, rng);
    const Y = Tensor.zeros(N, K);
    for (let i = 0; i < N; i++) Y.data[i * K + Math.floor(rng() * K)] = 1;
    crossEntropy(logits, Y).backward();
    for (let i = 0; i < N; i++) {
      let z = 0;
      for (let j = 0; j < K; j++) z += Math.exp(logits.data[i * K + j]);
      for (let j = 0; j < K; j++) {
        const s = Math.exp(logits.data[i * K + j]) / z;
        expect(logits.grad![i * K + j]).toBeCloseTo((s - Y.data[i * K + j]) / N, 14);
      }
    }
  });

  it("passes the end-to-end MLP check once no ReLU sits on its kink", () => {
    expect(mlpCheck(false)).toBeLessThan(TOLERANCE);
  });

  it("fails the MLP check with zero biases, because the origin puts ReLUs on their kink", () => {
    expect(mlpCheck(true)).toBeGreaterThan(1e-3);
  });

  it("trains an MLP to separate the spirals", () => {
    const data = spirals(100, 3, 1);
    const model = new MLP([2, 32, 32, 3], 1);
    for (let i = 0; i < 1500; i++) model.step(data, 0.5);
    expect(model.accuracy(data)).toBeGreaterThan(0.95);
  });
});
