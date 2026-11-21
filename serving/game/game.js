// Gap Glider: fly through the gaps. In the model modes, every few frames the game describes the situation
// in text and asks the decision model (serving/server.py) to choose between flapping and gliding.
"use strict";

const WORLD = { width: 420, height: 560, ground: 520 };
const PHYSICS = { gravity: 0.36, flapVelocity: -6.0, maxFall: 8.5, speed: 2.2 };
const GLIDER = { x: 110, radius: 13 };
const COLUMNS = { width: 62, gap: 170, spacing: 235, margin: 70 };
const DECISION_FRAMES = 4; // the model decides every 4 frames; the game waits for each answer
const PLAN_FRAMES = 18; // how far ahead the planner looks when it ranks the two actions
const SAFETY_FRAMES = 60; // an action is safe if some later choices avoid a crash for this long; 60 lets a random player survive
const COLORS = {
  bird: "#5ab4dc", wing: "#3d86ad", belly: "#dcecf3", beak: "#c5f82a", eye: "#0b0c0a",
  column: "#3b3f37", edge: "#6c7166", ground: "#2a2d26", ink: "#f3f3ef",
};

const $ = (id) => document.getElementById(id);
const canvas = $("game");
const ctx = canvas.getContext("2d");

const session = { key: null, running: false, waiting: false, frame: 0, decisions: 0, crashes: 0, best: 0, flapPose: 0, agreed: 0, interventions: 0 };
let world = newWorld();

function newWorld() {
  return { y: WORLD.height * 0.42, vy: 0, score: 0, columns: [newColumn(WORLD.width + 40)] };
}

function newColumn(x) {
  const lowest = WORLD.ground - COLUMNS.margin - COLUMNS.gap / 2;
  const highest = COLUMNS.margin + COLUMNS.gap / 2;
  return { x, center: highest + Math.random() * (lowest - highest), passed: false };
}

// --- Physics ---

function step(state, flap) {
  const vy = flap ? PHYSICS.flapVelocity : Math.min(state.vy + PHYSICS.gravity, PHYSICS.maxFall);
  return { ...state, vy, y: state.y + vy };
}

function hits(y, columns) {
  if (y + GLIDER.radius >= WORLD.ground || y - GLIDER.radius <= 0) return true;
  return columns.some((c) => GLIDER.x + GLIDER.radius > c.x && GLIDER.x - GLIDER.radius < c.x + COLUMNS.width
    && Math.abs(y - c.center) > COLUMNS.gap / 2 - GLIDER.radius);
}

function advance(flap) {
  Object.assign(world, step(world, flap));
  for (const c of world.columns) c.x -= PHYSICS.speed;
  const last = world.columns[world.columns.length - 1];
  if (last.x < WORLD.width - COLUMNS.spacing) world.columns.push(newColumn(last.x + COLUMNS.spacing));
  world.columns = world.columns.filter((c) => c.x + COLUMNS.width > -10);
  for (const c of world.columns) {
    if (!c.passed && c.x + COLUMNS.width < GLIDER.x) { c.passed = true; world.score += 1; }
  }
  session.flapPose = flap ? 8 : Math.max(0, session.flapPose - 1);
  if (hits(world.y, world.columns)) crash();
}

function crash() {
  session.crashes += 1;
  session.best = Math.max(session.best, world.score);
  world = newWorld();
  updateStats();
}

// --- What the model reads ---

function nextColumn() {
  return world.columns.find((c) => c.x + COLUMNS.width > GLIDER.x - GLIDER.radius);
}

function motionText(vy) {
  if (vy > 0.5) return `falling at ${vy.toFixed(1)} px per frame`;
  if (vy < -0.5) return `rising at ${(-vy).toFixed(1)} px per frame`;
  return "level";
}

function describe() {
  const c = nextColumn();
  const ahead = Math.max(0, Math.round(c.x - GLIDER.x));
  const offset = Math.round(c.center - world.y); // positive: the gap's center is lower on screen
  const where = offset === 0 ? "level with you" : `${Math.abs(offset)} px ${offset > 0 ? "below" : "above"} you`;
  const ground = Math.round(WORLD.ground - world.y - GLIDER.radius);
  return `Side-scrolling flying game. The next gap is ${ahead} px ahead, and its center is ${where}. `
    + `You are ${motionText(world.vy)}. The ground is ${ground} px below you.`;
}

// The planner, used by the hinted prompt, the agreement count and the shield, like the planner of CLM's game.
// An action is safe if, after it, some sequence of later decisions avoids a crash for SAFETY_FRAMES frames.
// Among safe actions, the best is the one that ends closer to the gap's center if the glider then stops flapping.
function plan() {
  const c = nextColumn();
  const rollout = (flap) => {
    let s = { y: world.y, vy: world.vy };
    for (let f = 1; f <= PLAN_FRAMES; f++) s = step(s, flap && f === 1);
    return { error: Math.abs(s.y - c.center), moved: Math.round(s.y - world.y) };
  };
  const options = {};
  for (const name of ["flap", "glide"]) {
    const after = window_(world, 0, name === "flap");
    options[name] = { ...rollout(name === "flap"), safe: after !== null && survives(after, DECISION_FRAMES) };
  }
  const safeNames = ["flap", "glide"].filter((n) => options[n].safe);
  const candidates = safeNames.length ? safeNames : ["flap", "glide"];
  const best = candidates.reduce((a, b) => (options[a].error <= options[b].error ? a : b));
  return { ...options, best };
}

// One decision window: the action on its first frame, then gliding. Null if the glider crashes in it.
function window_(state, frame, flap) {
  let s = { y: state.y, vy: state.vy };
  for (let f = 0; f < DECISION_FRAMES; f++) {
    s = step(s, flap && f === 0);
    if (hits(s.y, shifted(world.columns, frame + f + 1))) return null;
  }
  return s;
}

function survives(state, frame) {
  if (frame >= SAFETY_FRAMES) return true;
  return [false, true].some((flap) => {
    const next = window_(state, frame, flap);
    return next !== null && survives(next, frame + DECISION_FRAMES);
  });
}

function shifted(columns, frames) {
  return columns.map((c) => ({ ...c, x: c.x - PHYSICS.speed * frames }));
}

function question(mode, p) {
  const instructions = "Choose the action that keeps the flyer level with the center of the gap.";
  if (mode === "plain") {
    return { instructions, criteria: { flap: "Flap: jump upward, rising quickly.", glide: "Glide: do nothing and keep falling." } };
  }
  const label = (name, verb, o) => `${o.safe ? "Safe" : "Unsafe"}. ${verb}, then ${o.moved < 0 ? "rises" : "drops"} ${Math.abs(o.moved)} px.`
    + (o.safe ? "" : " Collision.") + (p.best === name ? " Best." : "");
  return { instructions, criteria: { flap: label("flap", "Flaps", p.flap), glide: label("glide", "Glides", p.glide) } };
}

// --- Asking the model ---

// Returns the action to execute: the model's proposal, unless the shield replaces an unsafe one.
async function decide(mode) {
  const state = describe();
  const p = plan();
  const q = question(mode, p);
  $("prompt").textContent = `${state}\n\n${q.instructions}\n- flap: ${q.criteria.flap}\n- glide: ${q.criteria.glide}`;
  const started = performance.now();
  const slow = setTimeout(() => status("Waiting for the GPU. The first request can take several minutes while it starts."), 3000);
  try {
    const response = await fetch(`${endpoint()}/v1/systemone`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${session.key}` },
      body: JSON.stringify({ state, questions: { action: { type: "choice", instructions: q.instructions, criteria: q.criteria } } }),
    });
    if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
    const body = await response.json();
    const probabilities = body.answers.action.probabilities;
    showDecision(probabilities, response.headers.get("X-CLM-Latency-Ms"), performance.now() - started);
    status("");
    const proposed = probabilities.flap > probabilities.glide ? "flap" : "glide";
    const other = proposed === "flap" ? "glide" : "flap";
    if (proposed === p.best) session.agreed += 1; // counted before the shield, as in CLM's game
    if ($("shield").checked && !p[proposed].safe && p[other].safe) {
      session.interventions += 1;
      return other === "flap";
    }
    return proposed === "flap";
  } finally {
    clearTimeout(slow);
  }
}

function endpoint() {
  return $("endpoint").value.trim().replace(/\/+$/, "");
}

// --- Loop ---

const pressed = { flap: false };

async function tick() {
  if (!session.running) return;
  const mode = $("mode").value;
  if (mode === "human") {
    advance(pressed.flap);
    pressed.flap = false;
  } else if (!session.waiting) {
    let flap = false;
    if (session.frame % DECISION_FRAMES === 0) {
      session.waiting = true;
      try {
        flap = await decide(mode);
        session.decisions += 1;
      } catch (error) {
        status(`Request failed: ${error.message}`, true);
        session.running = false;
      }
      session.waiting = false;
    }
    if (session.running) advance(flap);
  }
  session.frame += 1;
  draw();
  updateStats();
  requestAnimationFrame(tick);
}

// --- Drawing ---

function draw() {
  ctx.clearRect(0, 0, WORLD.width, WORLD.height);
  for (const c of world.columns) {
    const top = c.center - COLUMNS.gap / 2, bottom = c.center + COLUMNS.gap / 2;
    ctx.fillStyle = COLORS.column;
    ctx.fillRect(c.x, 0, COLUMNS.width, top);
    ctx.fillRect(c.x, bottom, COLUMNS.width, WORLD.ground - bottom);
    ctx.fillStyle = COLORS.edge;
    ctx.fillRect(c.x - 3, top - 10, COLUMNS.width + 6, 10);
    ctx.fillRect(c.x - 3, bottom, COLUMNS.width + 6, 10);
  }
  ctx.fillStyle = COLORS.ground;
  ctx.fillRect(0, WORLD.ground, WORLD.width, WORLD.height - WORLD.ground);
  drawGlider();
  ctx.fillStyle = COLORS.ink;
  ctx.font = "bold 34px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText(String(world.score), WORLD.width / 2, 60);
}

// A small bird seen from the side, facing right: forked tail, body, head, beak, and a wing that beats.
function drawGlider() {
  const tilt = Math.max(-0.5, Math.min(0.8, world.vy / 10));
  const wingUp = session.flapPose > 4;
  ctx.save();
  ctx.translate(GLIDER.x, world.y);
  ctx.rotate(tilt);
  shape(COLORS.bird, [[-13, -2], [-25, -9], [-20, 0], [-25, 8], [-13, 3]]); // tail
  ctx.fillStyle = COLORS.bird;
  ctx.beginPath();
  ctx.ellipse(0, 1, 15, 10, 0, 0, Math.PI * 2); // body
  ctx.arc(11, -6, 7.5, 0, Math.PI * 2); // head
  ctx.fill();
  ctx.fillStyle = COLORS.belly;
  ctx.beginPath();
  ctx.ellipse(2, 6, 9, 4, 0, 0, Math.PI * 2);
  ctx.fill();
  shape(COLORS.beak, [[17, -9], [27, -5], [17, -3]]);
  shape(COLORS.wing, wingUp ? [[-6, -2], [4, -2], [-4, -19]] : [[-7, 0], [5, 0], [-3, 13]]);
  ctx.fillStyle = COLORS.eye;
  ctx.beginPath();
  ctx.arc(13, -8, 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function shape(color, points) {
  ctx.fillStyle = color;
  ctx.beginPath();
  points.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  ctx.closePath();
  ctx.fill();
}

// --- Page ---

function showDecision(probabilities, serverMs, tripMs) {
  for (const name of ["flap", "glide"]) {
    const bar = $(`bar-${name}`);
    const p = probabilities[name];
    bar.children[1].style.width = `${(p * 100).toFixed(0)}%`;
    bar.children[2].textContent = p.toFixed(2);
    bar.classList.toggle("chosen", p === Math.max(probabilities.flap, probabilities.glide));
  }
  $("server-ms").textContent = serverMs ? Number(serverMs).toFixed(0) : "–";
  $("trip-ms").textContent = tripMs.toFixed(0);
}

function updateStats() {
  $("score").textContent = world.score;
  $("best").textContent = Math.max(session.best, world.score);
  $("crashes").textContent = session.crashes;
  $("decisions").textContent = session.decisions;
  $("agreement").textContent = session.decisions ? `${Math.round((100 * session.agreed) / session.decisions)}%` : "–";
  $("interventions").textContent = session.interventions;
}

function status(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

async function loadKey() {
  const response = await fetch("session.json", { cache: "no-store" });
  if (!response.ok) throw new Error("session.json not found: start the server once with modal serve -m serving.server");
  session.key = (await response.json()).key;
}

$("start").addEventListener("click", async () => {
  if (session.running) return;
  localStorage.setItem("gap-glider-endpoint", endpoint());
  if ($("mode").value !== "human") {
    if (!endpoint()) return status("Paste the endpoint URL first.", true);
    try { await loadKey(); } catch (error) { return status(error.message, true); }
  }
  status("");
  session.running = true;
  requestAnimationFrame(tick);
});
$("stop").addEventListener("click", () => { session.running = false; });
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && $("mode").value === "human") { pressed.flap = true; event.preventDefault(); }
});
canvas.addEventListener("pointerdown", () => { if ($("mode").value === "human") pressed.flap = true; });

$("endpoint").value = new URLSearchParams(location.search).get("endpoint") || localStorage.getItem("gap-glider-endpoint") || "";
draw();
