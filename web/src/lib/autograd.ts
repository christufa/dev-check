/**
 * A minimal reverse-mode autograd over 2-D float64 tensors.
 *
 * TypeScript port of the NumPy engine in the backprop lesson. Every tensor is rows × cols (a scalar is 1 × 1).
 * Elementwise ops broadcast NumPy-style along any dimension of size 1; their VJPs sum the gradient back down,
 * because broadcasting and summation are adjoint.
 */

type Vjp = (g: Float64Array) => Float64Array[];

export class Tensor {
  grad: Float64Array | null = null;
  vjp: Vjp = () => [];

  constructor(
    readonly data: Float64Array,
    readonly rows: number,
    readonly cols: number,
    readonly parents: Tensor[] = [],
    readonly op = "",
  ) {
    if (data.length !== rows * cols) throw new Error(`data has ${data.length} entries, expected ${rows}×${cols}`);
  }

  static scalar(x: number): Tensor {
    return new Tensor(Float64Array.of(x), 1, 1);
  }

  static from(rows: number[][]): Tensor {
    const r = rows.length;
    const c = rows[0].length;
    return new Tensor(Float64Array.from(rows.flat()), r, c);
  }

  static zeros(rows: number, cols: number): Tensor {
    return new Tensor(new Float64Array(rows * cols), rows, cols);
  }

  static randn(rows: number, cols: number, std: number, rng: () => number): Tensor {
    const d = new Float64Array(rows * cols);
    for (let i = 0; i < d.length; i++) d[i] = std * gaussian(rng);
    return new Tensor(d, rows, cols);
  }

  get size(): number {
    return this.data.length;
  }

  item(): number {
    if (this.size !== 1) throw new Error("item() needs a 1×1 tensor");
    return this.data[0];
  }

  /** Reverse-mode sweep: fills .grad on every node in this scalar's graph. */
  backward(): void {
    if (this.size !== 1) throw new Error("backward() needs a scalar output");
    const order = topo(this);
    for (const t of order) t.grad = new Float64Array(t.size);
    this.grad![0] = 1;
    for (let k = order.length - 1; k >= 0; k--) {
      const t = order[k];
      const gs = t.vjp(t.grad!);
      t.parents.forEach((p, i) => {
        const pg = p.grad!;
        const g = gs[i];
        for (let j = 0; j < pg.length; j++) pg[j] += g[j]; // fan-out: contributions accumulate
      });
    }
  }

  // --- elementwise binary ops (broadcasting) ---
  add(o: Tensor | number): Tensor {
    return binary(this, lift(o), (a, b) => a + b, () => 1, () => 1, "+");
  }
  sub(o: Tensor | number): Tensor {
    return binary(this, lift(o), (a, b) => a - b, () => 1, () => -1, "-");
  }
  mul(o: Tensor | number): Tensor {
    return binary(this, lift(o), (a, b) => a * b, (_, b) => b, (a) => a, "*");
  }
  div(o: Tensor | number): Tensor {
    return binary(this, lift(o), (a, b) => a / b, (_, b) => 1 / b, (a, b) => -a / (b * b), "/");
  }

  // --- elementwise unary ops: G_X = G ⊙ φ'(X) ---
  neg(): Tensor {
    return unary(this, (x) => -x, () => -1, "neg");
  }
  pow(p: number): Tensor {
    return unary(this, (x) => x ** p, (x) => p * x ** (p - 1), `**${p}`);
  }
  exp(): Tensor {
    return unary(this, Math.exp, (_, y) => y, "exp");
  }
  log(): Tensor {
    return unary(this, Math.log, (x) => 1 / x, "log");
  }
  tanh(): Tensor {
    return unary(this, Math.tanh, (_, y) => 1 - y * y, "tanh");
  }
  relu(): Tensor {
    return unary(this, (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0), "relu");
  }

  /** C = AB.  VJPs: G_A = G Bᵀ, G_B = Aᵀ G. */
  matmul(o: Tensor): Tensor {
    const a = this;
    const b = o;
    if (a.cols !== b.rows) throw new Error(`matmul shapes ${a.rows}×${a.cols} @ ${b.rows}×${b.cols}`);
    const out = node(matmulRaw(a.data, b.data, a.rows, a.cols, b.cols), a.rows, b.cols, [a, b], "@");
    out.vjp = (g) => [
      matmulRaw(g, transpose(b.data, b.rows, b.cols), a.rows, b.cols, b.rows),
      matmulRaw(transpose(a.data, a.rows, a.cols), g, a.cols, a.rows, b.cols),
    ];
    return out;
  }

  /** Sum over everything (axis undefined → 1×1), over rows (axis 0 → 1×cols) or over columns (axis 1 → rows×1). */
  sum(axis?: 0 | 1): Tensor {
    const a = this;
    const [r, c] = axis === undefined ? [1, 1] : axis === 0 ? [1, a.cols] : [a.rows, 1];
    const d = new Float64Array(r * c);
    for (let i = 0; i < a.rows; i++)
      for (let j = 0; j < a.cols; j++) d[(r === 1 ? 0 : i) * c + (c === 1 ? 0 : j)] += a.data[i * a.cols + j];
    const out = node(d, r, c, [a], "sum");
    // the VJP of a sum is a broadcast
    out.vjp = (g) => {
      const ga = new Float64Array(a.size);
      for (let i = 0; i < a.rows; i++)
        for (let j = 0; j < a.cols; j++) ga[i * a.cols + j] = g[(r === 1 ? 0 : i) * c + (c === 1 ? 0 : j)];
      return [ga];
    };
    return out;
  }

  mean(axis?: 0 | 1): Tensor {
    const s = this.sum(axis);
    return s.mul(s.size / this.size);
  }

  /** Row-wise log-softmax.  VJP per row: g_x = g − (1ᵀg) s. */
  logSoftmax(): Tensor {
    const a = this;
    const { rows: r, cols: c } = a;
    const y = new Float64Array(a.size);
    const s = new Float64Array(a.size);
    for (let i = 0; i < r; i++) {
      let m = -Infinity;
      for (let j = 0; j < c; j++) m = Math.max(m, a.data[i * c + j]);
      let z = 0;
      for (let j = 0; j < c; j++) z += Math.exp(a.data[i * c + j] - m);
      const lse = m + Math.log(z); // stable log-sum-exp
      for (let j = 0; j < c; j++) {
        y[i * c + j] = a.data[i * c + j] - lse;
        s[i * c + j] = Math.exp(y[i * c + j]);
      }
    }
    const out = node(y, r, c, [a], "log_softmax");
    out.vjp = (g) => {
      const ga = new Float64Array(a.size);
      for (let i = 0; i < r; i++) {
        let total = 0;
        for (let j = 0; j < c; j++) total += g[i * c + j];
        for (let j = 0; j < c; j++) ga[i * c + j] = g[i * c + j] - s[i * c + j] * total;
      }
      return [ga];
    };
    return out;
  }
}

/** Mean cross-entropy of row-wise logits against one-hot labels Y. */
export function crossEntropy(logits: Tensor, Y: Tensor): Tensor {
  return logits.logSoftmax().mul(Y).sum().mul(-1 / logits.rows);
}

function lift(x: Tensor | number): Tensor {
  return typeof x === "number" ? Tensor.scalar(x) : x;
}

function node(data: Float64Array, rows: number, cols: number, parents: Tensor[], op: string): Tensor {
  return new Tensor(data, rows, cols, parents, op);
}

function bdim(m: number, n: number): number {
  if (m === n || n === 1) return m;
  if (m === 1) return n;
  throw new Error(`cannot broadcast ${m} with ${n}`);
}

function binary(
  a: Tensor,
  b: Tensor,
  f: (a: number, b: number) => number,
  da: (a: number, b: number) => number,
  db: (a: number, b: number) => number,
  op: string,
): Tensor {
  const R = bdim(a.rows, b.rows);
  const C = bdim(a.cols, b.cols);
  const ia = (i: number, j: number) => (a.rows === 1 ? 0 : i) * a.cols + (a.cols === 1 ? 0 : j);
  const ib = (i: number, j: number) => (b.rows === 1 ? 0 : i) * b.cols + (b.cols === 1 ? 0 : j);
  const d = new Float64Array(R * C);
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) d[i * C + j] = f(a.data[ia(i, j)], b.data[ib(i, j)]);
  const out = node(d, R, C, [a, b], op);
  out.vjp = (g) => {
    const ga = new Float64Array(a.size);
    const gb = new Float64Array(b.size);
    for (let i = 0; i < R; i++)
      for (let j = 0; j < C; j++) {
        const x = a.data[ia(i, j)];
        const y = b.data[ib(i, j)];
        const gij = g[i * C + j];
        ga[ia(i, j)] += gij * da(x, y); // += sums the gradient back over broadcast dimensions
        gb[ib(i, j)] += gij * db(x, y);
      }
    return [ga, gb];
  };
  return out;
}

function unary(a: Tensor, f: (x: number) => number, df: (x: number, y: number) => number, op: string): Tensor {
  const d = a.data.map(f);
  const out = node(d, a.rows, a.cols, [a], op);
  out.vjp = (g) => [g.map((gi, i) => gi * df(a.data[i], d[i]))];
  return out;
}

export function matmulRaw(a: Float64Array, b: Float64Array, m: number, k: number, n: number): Float64Array {
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++)
    for (let p = 0; p < k; p++) {
      const aip = a[i * k + p];
      if (aip === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += aip * b[p * n + j];
    }
  return out;
}

function transpose(a: Float64Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[j * rows + i] = a[i * cols + j];
  return out;
}

/** Iterative post-order DFS: every node appears after all of its parents. */
function topo(root: Tensor): Tensor[] {
  const order: Tensor[] = [];
  const seen = new Set<Tensor>();
  const stack: [Tensor, boolean][] = [[root, false]];
  while (stack.length) {
    const [t, expanded] = stack.pop()!;
    if (expanded) {
      order.push(t);
      continue;
    }
    if (seen.has(t)) continue;
    seen.add(t);
    stack.push([t, true]);
    for (const p of t.parents) if (!seen.has(p)) stack.push([p, false]);
  }
  return order;
}

// --- randomness, gradient checking ---

/** Seeded PRNG (mulberry32) so demos are reproducible. */
export function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: () => number): number {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); // Box–Muller
}

/** Central differences of f() with respect to every entry of x, perturbed in place. */
export function numericalGrad(f: () => number, x: Float64Array, h = 1e-6): Float64Array {
  const g = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const old = x[i];
    x[i] = old + h;
    const fp = f();
    x[i] = old - h;
    const fm = f();
    x[i] = old;
    g[i] = (fp - fm) / (2 * h);
  }
  return g;
}

/**
 * Compares backward() with central differences for every input.
 * Returns the worst relative error ‖g_auto − g_num‖ / (‖g_auto‖ + ‖g_num‖) over the inputs.
 */
export function gradCheck(build: (...xs: Tensor[]) => Tensor, ...inputs: Tensor[]): number {
  const leaves = inputs.map((t) => new Tensor(t.data.slice(), t.rows, t.cols));
  build(...leaves).backward();
  let worst = 0;
  for (const leaf of leaves) {
    const num = numericalGrad(() => build(...leaves.map((l) => new Tensor(l.data, l.rows, l.cols))).item(), leaf.data);
    const auto = leaf.grad!;
    let diff = 0;
    let na = 0;
    let nn = 0;
    for (let i = 0; i < num.length; i++) {
      diff += (auto[i] - num[i]) ** 2;
      na += auto[i] ** 2;
      nn += num[i] ** 2;
    }
    worst = Math.max(worst, Math.sqrt(diff) / (Math.sqrt(na) + Math.sqrt(nn) + 1e-300));
  }
  return worst;
}
