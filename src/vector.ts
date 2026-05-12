const DIMS = 14;
const K = 3;

async function loadBinary(path: string): Promise<ArrayBuffer> {
  try {
    return await Bun.file(path).arrayBuffer();
  } catch (e) {
    console.error(`Failed to load binary file "${path}":`, e);
    process.exit(1);
  }
}

// Carrega os 4 binários em paralelo
const [vBuf, lBuf, treeBuf] = await Promise.all([
  loadBinary("vectors.bin"),
  loadBinary("labels.bin"),
  loadBinary("vptree.bin"),
]);

const referenceVectors = new Float32Array(vBuf);
const referenceLabels = new Uint8Array(lBuf);
export const referenceCount = referenceLabels.length;

// Leitura das 4 seções do VP-Tree via views sobre o mesmo buffer
const vpIndices = new Int32Array(treeBuf, 0, referenceCount);
const vpMus = new Float32Array(treeBuf, referenceCount * 4, referenceCount);
const vpLefts = new Int32Array(treeBuf, referenceCount * 8, referenceCount);
const vpRights = new Int32Array(treeBuf, referenceCount * 12, referenceCount);

// ── Buffers reutilizáveis por requisição (seguro: Bun é single-threaded) ──
export const queryVector = new Float32Array(DIMS); // escrito pelo index.ts

// Stack de traversal do VP-Tree: profundidade máxima ≈ 2 × log₂(N)
// Para N = 1M: ≈ 40; reservamos 1024 para folga.
const searchStack = new Int32Array(1024);

// Max-heap de tamanho fixo K para os top-K vizinhos
const heapDists = new Float32Array(K).fill(Infinity);
const heapLabels = new Uint8Array(K);

// ── Helpers inline ────────────────────────────────────────────────────────

function euclidean(refIdx: number): number {
  let sum = 0;
  const base = refIdx * DIMS;
  for (let j = 0; j < DIMS; j++) {
    const d = queryVector[j]! - referenceVectors[base + j]!;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// Insere no max-heap de tamanho K. Retorna o novo tau (máximo atual).
function heapInsert(dist: number, label: number, heapSize: number): number {
  if (heapSize < K) {
    heapDists[heapSize] = dist;
    heapLabels[heapSize] = label;
    return dist; // tau temporário (não vale buscar pelo max ainda)
  }
  // Heap já cheio: encontrar o pior slot
  let maxDist = heapDists[0]!;
  let maxIdx = 0;
  for (let i = 1; i < K; i++) {
    if (heapDists[i]! > maxDist) {
      maxDist = heapDists[i]!;
      maxIdx = i;
    }
  }
  if (dist < maxDist) {
    heapDists[maxIdx] = dist;
    heapLabels[maxIdx] = label;
    // Recalcular novo tau
    let newMax = heapDists[0]!;
    for (let i = 1; i < K; i++) {
      if (heapDists[i]! > newMax) newMax = heapDists[i]!;
    }
    return newMax;
  }
  return maxDist;
}

// ── k-NN via VP-Tree ──────────────────────────────────────────────────────

export function knnFraudScore(): number {
  // Resetar heap
  heapDists.fill(Infinity);

  let stackTop = 0;
  let heapSize = 0;
  let tau = Infinity;

  searchStack[stackTop++] = 0; // começa pela raiz (nodeIdx = 0)

  while (stackTop > 0) {
    const nodeIdx = searchStack[--stackTop]!;
    const refIdx = vpIndices[nodeIdx]!;

    const d = euclidean(refIdx);

    if (d < tau || heapSize < K) {
      tau = heapInsert(d, referenceLabels[refIdx]!, heapSize);
      if (heapSize < K) heapSize++;
    }

    const mu = vpMus[nodeIdx]!;
    const left = vpLefts[nodeIdx]!;
    const right = vpRights[nodeIdx]!;

    // Pruning por desigualdade triangular:
    // Entrar na esquerda se d - tau < mu  (bola interna pode ter ponto dentro do raio tau)
    // Entrar na direita  se d + tau >= mu (bola externa pode ter ponto dentro do raio tau)
    const enterLeft = left >= 0 && d - tau < mu;
    const enterRight = right >= 0 && d + tau >= mu;

    // Empilhar o ramo menos provável primeiro (processado por último)
    // se d < mu, esquerda é mais próxima → empilha direita antes, esquerda depois
    if (d < mu) {
      if (enterRight) searchStack[stackTop++] = right;
      if (enterLeft) searchStack[stackTop++] = left;
    } else {
      if (enterLeft) searchStack[stackTop++] = left;
      if (enterRight) searchStack[stackTop++] = right;
    }
  }

  // Weighted fraud score: soma dos pesos dos vizinhos com label=fraud
  let fraudWeight = 0;
  let totalWeight = 0;
  for (let i = 0; i < K; i++) {
    const w = 1 / (heapDists[i]! + 1e-6);
    totalWeight += w;
    if (heapLabels[i] === 1) fraudWeight += w;
  }

  return fraudWeight / totalWeight;
}
