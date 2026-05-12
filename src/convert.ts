import { write } from "bun";

const DIMS = 14;

const file = Bun.file("references.json");
const data: { vector: number[]; label: string }[] = await file.json();
const total = data.length;

// --- Vetores e labels ---
const vectorBuffer = new Float32Array(total * DIMS);
const labelBuffer = new Uint8Array(total);

for (let i = 0; i < total; i++) {
  vectorBuffer.set(data[i]!.vector, i * DIMS);
  labelBuffer[i] = data[i]!.label === "legit" ? 0 : 1;
}

await write("vectors.bin", Buffer.from(vectorBuffer.buffer));
await write("labels.bin", Buffer.from(labelBuffer.buffer));
console.log(`Vetores: ${total} elementos convertidos.`);

// --- VP-Tree ---
// Layout de vptree.bin (N × 16 bytes, 4 seções de N × 4 bytes cada):
//   [0      .. N×4-1 ] vpIndices : Int32   — qual vetor de referência este nó representa
//   [N×4   .. N×8-1  ] vpMus     : Float32 — threshold (mediana das distâncias)
//   [N×8   .. N×12-1 ] vpLefts   : Int32   — filho esquerdo (-1 = nenhum)
//   [N×12  .. N×16-1 ] vpRights  : Int32   — filho direito  (-1 = nenhum)

const vpIndices = new Int32Array(total);
const vpMus = new Float32Array(total);
const vpLefts = new Int32Array(total).fill(-1);
const vpRights = new Int32Array(total).fill(-1);

let nodeCount = 0;

function euclidean(a: number, b: number): number {
  let sum = 0;
  const ao = a * DIMS;
  const bo = b * DIMS;
  for (let j = 0; j < DIMS; j++) {
    const d = vectorBuffer[ao + j]! - vectorBuffer[bo + j]!;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// Índices de trabalho usados in-place durante o build
const workIndices = new Int32Array(total);
for (let i = 0; i < total; i++) workIndices[i] = i;

// Build iterativo com stack explícita (evita estouro de pilha de recursão para N grande)
type Frame = { lo: number; hi: number; parent: number; isLeft: boolean };
const buildStack: Frame[] = [{ lo: 0, hi: total - 1, parent: -1, isLeft: true }];

while (buildStack.length > 0) {
  const frame = buildStack.pop()!;
  const { lo, hi, parent, isLeft } = frame;

  if (lo > hi) continue;

  const nodeIdx = nodeCount++;
  if (parent >= 0) {
    if (isLeft) vpLefts[parent] = nodeIdx;
    else vpRights[parent] = nodeIdx;
  }

  const vp = workIndices[lo]!;
  vpIndices[nodeIdx] = vp;

  if (lo === hi) {
    vpMus[nodeIdx] = 0;
    continue;
  }

  const n = hi - lo;

  // Distâncias do VP para os demais itens no intervalo [lo+1..hi]
  const dists = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dists[i] = euclidean(vp, workIndices[lo + 1 + i]!);
  }

  // Mediana das distâncias → threshold do nó
  const sortedDists = dists.slice().sort();
  const mu = sortedDists[Math.floor(n / 2)]!;
  vpMus[nodeIdx] = mu;

  // Particionar workIndices[lo+1..hi]: esquerda (dist < mu), direita (dist >= mu)
  const leftBuf: number[] = [];
  const rightBuf: number[] = [];
  for (let i = 0; i < n; i++) {
    if (dists[i]! < mu) leftBuf.push(workIndices[lo + 1 + i]!);
    else rightBuf.push(workIndices[lo + 1 + i]!);
  }

  let pos = lo + 1;
  for (const idx of leftBuf) workIndices[pos++] = idx;
  const leftHi = pos - 1;
  for (const idx of rightBuf) workIndices[pos++] = idx;

  // Empilhar direito antes para que esquerdo seja processado primeiro
  if (rightBuf.length > 0) {
    buildStack.push({ lo: leftHi + 1, hi, parent: nodeIdx, isLeft: false });
  }
  if (leftBuf.length > 0) {
    buildStack.push({ lo: lo + 1, hi: leftHi, parent: nodeIdx, isLeft: true });
  }
}

// Serializar em buffer único contíguo (4 seções × N × 4 bytes)
const treeBuf = new ArrayBuffer(total * 16);
new Int32Array(treeBuf, 0, total).set(vpIndices);
new Float32Array(treeBuf, total * 4, total).set(vpMus);
new Int32Array(treeBuf, total * 8, total).set(vpLefts);
new Int32Array(treeBuf, total * 12, total).set(vpRights);

await write("vptree.bin", Buffer.from(treeBuf));
console.log(`VP-Tree: ${nodeCount} nós serializados (${(total * 16 / 1024).toFixed(1)} KB).`);
