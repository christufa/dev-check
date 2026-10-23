import { Tensor, crossEntropy, gaussian, makeRng } from "./autograd";

export interface Dataset {
  X: Tensor;
  Y: Tensor;
  labels: number[];
  classes: number;
}

/** Interleaved spirals in [-1, 1]²: no linear model can separate the classes. */
export function spirals(perClass: number, classes: number, seed: number, noise = 0.2): Dataset {
  const rng = makeRng(seed);
  const xs: number[][] = [];
  const labels: number[] = [];
  for (let k = 0; k < classes; k++) {
    for (let i = 0; i < perClass; i++) {
      const r = i / (perClass - 1);
      const t = 4 * k + (4 * i) / (perClass - 1) + noise * gaussian(rng);
      xs.push([r * Math.sin(t), r * Math.cos(t)]);
      labels.push(k);
    }
  }
  return { X: Tensor.from(xs), Y: oneHot(labels, classes), labels, classes };
}

export function oneHot(labels: number[], classes: number): Tensor {
  const Y = Tensor.zeros(labels.length, classes);
  labels.forEach((k, i) => (Y.data[i * classes + k] = 1));
  return Y;
}

/** ReLU MLP with He initialization; each bias is a separate 1×d node, broadcast over the batch. */
export class MLP {
  params: Tensor[] = [];

  constructor(sizes: number[], seed: number) {
    const rng = makeRng(seed);
    for (let l = 0; l + 1 < sizes.length; l++) {
      this.params.push(Tensor.randn(sizes[l], sizes[l + 1], Math.sqrt(2 / sizes[l]), rng));
      this.params.push(Tensor.zeros(1, sizes[l + 1]));
    }
  }

  forward(X: Tensor, params: Tensor[] = this.params): Tensor {
    let h = X;
    const layers = params.length / 2;
    for (let l = 0; l < layers; l++) {
      h = h.matmul(params[2 * l]).add(params[2 * l + 1]);
      if (l < layers - 1) h = h.relu();
    }
    return h;
  }

  /** One full-batch gradient-descent step; returns the loss before the step. */
  step(data: Dataset, lr: number): number {
    const loss = crossEntropy(this.forward(data.X), data.Y);
    loss.backward();
    for (const p of this.params) {
      const g = p.grad!;
      for (let i = 0; i < p.size; i++) p.data[i] -= lr * g[i];
    }
    return loss.item();
  }

  predict(X: Tensor): number[] {
    const logits = this.forward(X);
    const out: number[] = [];
    for (let i = 0; i < logits.rows; i++) {
      let best = 0;
      for (let j = 1; j < logits.cols; j++)
        if (logits.data[i * logits.cols + j] > logits.data[i * logits.cols + best]) best = j;
      out.push(best);
    }
    return out;
  }

  accuracy(data: Dataset): number {
    const pred = this.predict(data.X);
    return pred.filter((p, i) => p === data.labels[i]).length / pred.length;
  }
}
