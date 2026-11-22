// Push-optimal Sokoban solver for make.fun sokoban-solver-01 (MIT, see LICENSE).
// The level is split into rooms (the hallway row is dead for boxes, so each room is an independent problem); each room
// is solved by A* over push states: Zobrist-hashed (boxes + normalized player region), a consistent admissible
// heuristic (minimum-cost box/goal matching on true push distances), and sound deadlock pruning (dead squares, 2x2
// blocks, freeze deadlocks). Work is bounded by node budgets, never the clock, so every run answers identically.
import { chmodSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const ROOM_NODES = 50_000; // expansions per room search (generated rooms need about 1,200 at most)
const LETTERS = 'udlr';

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

function parse(text: string): Level {
  const rows = text.split('\n').map((r) => r.replace(/\r$/, '')).filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#' || x === 0 || y === 0 || x === w - 1 || y === h - 1) continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

const dirsOf = (w: number) => [-w, w, -1, 1];

/** Push distance from every square to goal g (pulling the box back from g), ignoring other boxes. */
function pushDist(lv: Level, g: number): Int32Array {
  const { wall, w } = lv;
  const D = dirsOf(w);
  const dist = new Int32Array(wall.length).fill(-1);
  const q = [g];
  dist[g] = 0;
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    for (const d of D) {
      const y = x - d; // box at y pushed by d lands on x; player stands at y - d
      if (!wall[y] && !wall[y - d] && dist[y] < 0) {
        dist[y] = dist[x] + 1;
        q.push(y);
      }
    }
  }
  return dist;
}

/** Splits the level into independent rooms: the player's row squares that are dead for boxes are cut out. */
function roomsOf(lv: Level): { goals: number[]; boxes: number[] }[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const live = new Uint8Array(wall.length);
  for (const g of goals) {
    const d = pushDist(lv, g);
    for (let i = 0; i < d.length; i++) if (d[i] >= 0) live[i] = 1;
  }
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { goals: number[]; boxes: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const sq = [s];
    part[s] = rooms.length;
    for (let i = 0; i < sq.length; i++) {
      for (const d of dirsOf(w)) {
        const q = sq[i] + d;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          sq.push(q);
        }
      }
    }
    rooms.push({ goals: [], boxes: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Deterministic Zobrist tables (xorshift32 from a fixed seed), 26-bit halves combined into an exact 52-bit number key.
let seed = 0x9e3779b9;
const rnd = () => {
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  return (seed >>> 0) & 0x3ffffff;
};
/** Four Zobrist tables (box halves, player halves) for a board of N squares, the same on every run. */
const zobrist = (N: number): Int32Array[] => {
  seed = 0x9e3779b9;
  return [0, 1, 2, 3].map(() => Int32Array.from({ length: N }, rnd));
};

/**
 * A* over pushes on `lv` (weight 1: push-optimal). Returns the moves (lower = step, upper = push) or null when the
 * node budget runs out.
 */
function solveAstar(lv: Level, maxNodes: number, weight: number): { moves: string; pushes: number } | null {
  const { w, wall, goals } = lv;
  const N = wall.length;
  const D = dirsOf(w);
  const n = lv.boxes.length;
  const goal = new Uint8Array(N);
  for (const g of goals) goal[g] = 1;
  const dist = goals.map((g) => pushDist(lv, g));
  const live = new Uint8Array(N);
  for (const d of dist) for (let i = 0; i < N; i++) if (d[i] >= 0) live[i] = 1;
  if (lv.boxes.some((b) => !live[b])) return null;
  const INF = 1 << 20;
  // cost[b * n + j]: push distance from square b to goal j.
  const cost = new Int32Array(N * n);
  for (let c = 0; c < N; c++) for (let j = 0; j < n; j++) cost[c * n + j] = dist[j][c] < 0 ? INF : dist[j][c];

  // Heuristic: exact min-cost assignment by DP over goal subsets (n <= 8), else sum of nearest goals.
  const full = (1 << n) - 1;
  const dp = n <= 8 ? new Int32Array(1 << n) : null;
  const heur = (bs: number[]): number => {
    if (dp) {
      dp.fill(INF);
      dp[0] = 0;
      for (let m = 0; m < full; m++) {
        const v = dp[m];
        if (v >= INF) continue;
        let k = 0;
        for (let x = m; x; x &= x - 1) k++;
        const base = bs[k] * n;
        for (let j = 0; j < n; j++) {
          if (m & (1 << j)) continue;
          const c = v + cost[base + j];
          if (c < dp[m | (1 << j)]) dp[m | (1 << j)] = c;
        }
      }
      return dp[full];
    }
    let s = 0;
    for (const b of bs) {
      let m = INF;
      for (let j = 0; j < n; j++) if (cost[b * n + j] < m) m = cost[b * n + j];
      s += m;
    }
    return s;
  };

  const occ = new Uint8Array(N);
  const seen = new Int32Array(N);
  const seenC = new Int32Array(N);
  const stack = new Int32Array(N);
  let stamp = 0;
  /** Marks the player's region from p with a fresh stamp; returns its lowest square. */
  const reach = (p: number, seen: Int32Array): number => {
    stamp++;
    let low = p, sp = 0;
    stack[sp++] = p;
    seen[p] = stamp;
    while (sp > 0) {
      const x = stack[--sp];
      if (x < low) low = x;
      for (let k = 0; k < 4; k++) {
        const y = x + D[k];
        if (!wall[y] && !occ[y] && seen[y] !== stamp) {
          seen[y] = stamp;
          stack[sp++] = y;
        }
      }
    }
    return low;
  };

  // Freeze deadlock: a box blocked on both axes (walls, dead pairs or frozen boxes) off its goal.
  const tmpWall = new Uint8Array(N);
  let offGoal = false;
  const solid = (x: number) => wall[x] || tmpWall[x];
  const blocked = (b: number, axis: number): boolean => {
    const d = axis ? w : 1;
    const a = b - d, c = b + d;
    if (solid(a) || solid(c)) return true;
    if (!live[a] && !live[c]) return true;
    tmpWall[b] = 1;
    const r = (occ[a] && frozen(a, axis ^ 1)) || (occ[c] && frozen(c, axis ^ 1));
    tmpWall[b] = 0;
    return !!r;
  };
  const frozen = (b: number, axis: number): boolean => {
    // `axis` is the one still to check; the other is blocked by the caller (it is wall now).
    if (!blocked(b, axis)) return false;
    if (!goal[b]) offGoal = true;
    return true;
  };
  const freezeDead = (t: number): boolean => {
    offGoal = false;
    if (!(blocked(t, 0) && blocked(t, 1))) return false;
    return offGoal || !goal[t];
  };
  const block2x2 = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const sq = [a, a + 1, a + w, a + w + 1];
      if (!sq.every((x) => wall[x] || occ[x])) continue;
      if (sq.some((x) => occ[x] && !goal[x])) return true;
    }
    return false;
  };

  // State store.
  const SB: number[][] = [];
  const SP: number[] = [];
  const SG: number[] = [];
  const PAR: number[] = [];
  const SK: number[] = [];
  const index = new Map<number, number>();
  const open: number[][][] = [];
  let fmin = 0;
  const pushOpen = (s: number, f: number, g: number) => {
    (open[f] ??= [])[g] ??= [];
    open[f][g].push(s);
    if (f < fmin) fmin = f;
  };
  const [ZB1, ZB2, ZP1, ZP2] = zobrist(N);
  const hash = (bs: number[], low: number): number => {
    let a = ZP1[low], b = ZP2[low];
    for (const x of bs) (a ^= ZB1[x]), (b ^= ZB2[x]);
    return a * 0x4000000 + b;
  };

  const b0 = lv.boxes.slice().sort((x, y) => x - y);
  for (const b of b0) occ[b] = 1;
  const k0 = hash(b0, reach(lv.player, seenC));
  for (const b of b0) occ[b] = 0;
  SB.push(b0), SP.push(lv.player), SG.push(0), PAR.push(-1), SK.push(k0);
  index.set(k0, 0);
  const h0 = heur(b0);
  if (h0 >= INF) return null;
  pushOpen(0, Math.ceil(h0 * weight), 0);
  let found = -1;
  let expanded = 0;
  while (found < 0) {
    // Lowest f, then highest g (deepest first among ties), LIFO: deterministic.
    let s = -1;
    while (fmin < open.length) {
      const row = open[fmin];
      if (row) {
        for (let g = row.length - 1; g >= 0; g--) {
          const st = row[g];
          if (st && st.length) {
            s = st.pop()!;
            break;
          }
        }
      }
      if (s >= 0) break;
      fmin++;
    }
    if (s < 0) return null;
    if (index.get(SK[s]) !== s) continue;
    const bs = SB[s];
    if (bs.every((b) => goal[b])) {
      found = s;
      break;
    }
    if (++expanded > maxNodes) return null;
    for (const b of bs) occ[b] = 1;
    reach(SP[s], seen);
    const here = stamp;
    const g1 = SG[s] + 1;
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi];
      for (let k = 0; k < 4; k++) {
        const d = D[k];
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const ok = !block2x2(t) && !freezeDead(t);
        if (ok) {
          const next = bs.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const key = hash(next, reach(b, seenC));
          const prev = index.get(key);
          if (prev === undefined || SG[prev] > g1) {
            const id = SB.length;
            SB.push(next), SP.push(b), SG.push(g1), PAR.push(s), SK.push(key);
            index.set(key, id);
            const hv = heur(next);
            if (hv < INF) pushOpen(id, g1 + Math.ceil(hv * weight), g1);
          }
        }
        occ[t] = 0;
        occ[b] = 1;
      }
    }
    for (const b of bs) occ[b] = 0;
  }
  // Rebuild moves.
  const path: number[] = [];
  for (let s = found; s > 0; s = PAR[s]) path.push(s);
  path.reverse();
  let moves = '';
  let p = lv.player;
  let cur = b0;
  for (const s of path) {
    const nb = SB[s];
    const from = cur.find((b) => !nb.includes(b))!;
    const to = nb.find((b) => !cur.includes(b))!;
    const d = to - from;
    for (const b of cur) occ[b] = 1;
    moves += walk(lv, occ, p, from - d);
    for (const b of cur) occ[b] = 0;
    moves += LETTERS[D.indexOf(d)].toUpperCase();
    p = from;
    cur = nb;
  }
  return { moves, pushes: path.length };
}

/** Shortest walk from a to b avoiding walls and occupied squares (BFS, fixed direction order). */
function walk(lv: Level, occ: Uint8Array, a: number, b: number): string {
  if (a === b) return '';
  const { wall, w } = lv;
  const D = dirsOf(w);
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Int8Array(wall.length);
  const q = [a];
  prev[a] = a;
  for (let i = 0; i < q.length && prev[b] < 0; i++) {
    for (let k = 0; k < 4; k++) {
      const y = q[i] + D[k];
      if (wall[y] || occ[y] || prev[y] >= 0) continue;
      prev[y] = q[i];
      how[y] = k;
      q.push(y);
    }
  }
  if (prev[b] < 0) throw new Error('unreachable');
  let s = '';
  for (let x = b; x !== a; x = prev[x]) s = LETTERS[how[x]] + s;
  return s;
}

/** Replays moves; returns the push count when legal and solved, else -1. */
function verify(lv: Level, moves: string): number {
  const { w, wall } = lv;
  const box = new Uint8Array(wall.length);
  for (const b of lv.boxes) box[b] = 1;
  let p = lv.player, pushes = 0;
  for (const ch of moves) {
    const d = dirsOf(w)[LETTERS.indexOf(ch.toLowerCase())];
    const q = p + d;
    if (wall[q]) return -1;
    if (box[q]) {
      if (wall[q + d] || box[q + d]) return -1;
      box[q] = 0;
      box[q + d] = 1;
      pushes++;
    }
    p = q;
  }
  return lv.goals.every((g) => box[g]) ? pushes : -1;
}

function endOf(w: number, p: number, moves: string): number {
  for (const ch of moves) p += dirsOf(w)[LETTERS.indexOf(ch.toLowerCase())];
  return p;
}

function solve(text: string): string {
  const lv = parse(text);
  if (lv.player < 0 || lv.boxes.length === 0) return '';
  const rooms = roomsOf(lv);
  if (rooms) {
    let moves = '';
    let player = lv.player;
    const done = rooms.map(() => false);
    let ok = true;
    for (let i = 0; i < rooms.length && ok; i++) {
      // Other rooms' boxes stand as walls (on their goals once solved); the player walks on from where it stopped.
      const wall = lv.wall.slice();
      rooms.forEach((o, j) => j !== i && (done[j] ? o.goals : o.boxes).forEach((b) => (wall[b] = 1)));
      const sub: Level = { ...lv, wall, goals: rooms[i].goals.sort((a, b) => a - b), boxes: rooms[i].boxes.sort((a, b) => a - b), player };
      const r = solveAstar(sub, ROOM_NODES, 1) ?? solveAstar(sub, ROOM_NODES, 3);
      ok = r !== null;
      if (r) (moves += r.moves), (player = endOf(lv.w, player, r.moves)), (done[i] = true);
    }
    if (ok && verify(lv, moves) >= 0) return moves;
  }
  const r = solveAstar(lv, 100_000, 1) ?? solveAstar(lv, 100_000, 3);
  return r && verify(lv, r.moves) >= 0 ? r.moves : '';
}

mkdirSync(O, { recursive: true });
try {
  chmodSync(O, 0o755);
} catch {}
for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  const dir = join(O, id);
  mkdirSync(dir, { recursive: true });
  const file = join(dir, 'solution.txt');
  writeFileSync(file, `${moves}\n`);
  try {
    chmodSync(dir, 0o755);
    chmodSync(file, 0o644);
  } catch {}
}
