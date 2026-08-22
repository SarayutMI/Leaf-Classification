// src/static/app.js — rule-base admin page.
// Same origin as the API, so the session cookie rides along on every fetch.
"use strict";

const TRAIT_KEYS = ["shape", "apex", "base", "margin"];
const TRAIT_LABELS = {
  shape: "Shape (รูปร่างแผ่นใบ)",
  apex: "Apex (ปลายใบ)",
  base: "Base (โคนใบ)",
  margin: "Margin (ขอบใบ)",
};
const TRAIT_HINTS = {
  shape: "ทายจากภาพใบเต็มใบ",
  apex: "ปลายใบ — ทายจากแถบล่างของภาพ",
  base: "โคนใบ — ทายจากแถบบนของภาพ",
  margin: "ขอบใบ — ทายจากแถบกลางของภาพ",
};
let vocab = {};

const $ = (id) => document.getElementById(id);

// ── Busy state ────────────────────────────────────────────
// Buttons keep their width while busy so the layout does not jump.
function setBusy(button, label) {
  if (button.dataset.idleHtml === undefined) {
    button.dataset.idleHtml = button.innerHTML;
  }
  button.style.minWidth = `${button.offsetWidth}px`;
  button.disabled = true;
  button.classList.add("busy");
  button.innerHTML = `<span class="spinner" aria-hidden="true"></span>${label}`;
}

function clearBusy(button) {
  if (button.dataset.idleHtml !== undefined) {
    button.innerHTML = button.dataset.idleHtml;
  }
  button.disabled = false;
  button.classList.remove("busy");
  button.style.minWidth = "";
}

// ── API helper ────────────────────────────────────────────
class Unauthorized extends Error {}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: options.body ? { "Content-Type": "application/json" } : {},
    ...options,
  });

  if (response.status === 401) {
    showLogin();
    throw new Unauthorized("Not authenticated");
  }

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const details = body.errors && body.errors.details;
    throw new Error(details || body.message || `Request failed (${response.status})`);
  }
  return body.data || {};
}

function flash(message, isError = false) {
  const el = $("flash");
  el.textContent = message;
  el.classList.toggle("bad", isError);
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 4000);
}

function describeError(error) {
  if (error instanceof Unauthorized) return null;
  return error.message || "เกิดข้อผิดพลาด";
}

// ── Views ─────────────────────────────────────────────────
function showLogin() {
  $("boot-view").classList.add("hidden");
  $("login-view").classList.remove("hidden");
  $("app-view").classList.add("hidden");
  $("editor-backdrop").classList.add("hidden");
}

function showApp(username) {
  $("boot-view").classList.add("hidden");
  $("login-view").classList.add("hidden");
  $("app-view").classList.remove("hidden");
  $("current-user").textContent = username ? `ผู้ใช้: ${username}` : "";
}

// ── Login ─────────────────────────────────────────────────
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorEl = $("login-error");
  const button = $("login-submit");
  errorEl.classList.add("hidden");
  setBusy(button, "กำลังเข้าสู่ระบบ…");

  try {
    const data = await api("/api/admin/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("login-username").value,
        password: $("login-password").value,
      }),
    });
    $("login-password").value = "";
    showApp(data.username);
    await boot();
  } catch (error) {
    // api() flips to the login view on 401; here a 401 just means bad password.
    errorEl.textContent = error instanceof Unauthorized
      ? "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"
      : describeError(error);
    errorEl.classList.remove("hidden");
  } finally {
    clearBusy(button);
  }
});

$("logout-btn").addEventListener("click", async () => {
  const button = $("logout-btn");
  setBusy(button, "กำลังออก…");
  try {
    await fetch("/api/admin/logout", { method: "POST", credentials: "same-origin" });
  } finally {
    clearBusy(button);
    showLogin();
  }
});

// ── Tabs ──────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    $("tab-rules").classList.toggle("hidden", tab.dataset.tab !== "rules");
    $("tab-test").classList.toggle("hidden", tab.dataset.tab !== "test");
  });
});

// ── Trait inputs ──────────────────────────────────────────
function buildTraitSelectors() {
  const container = $("rule-traits");
  container.innerHTML = "";

  TRAIT_KEYS.forEach((key) => {
    const label = document.createElement("label");
    label.appendChild(document.createTextNode(TRAIT_LABELS[key]));

    const select = document.createElement("select");
    select.id = `rule-trait-${key}`;
    select.required = true;

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "— เลือก —";
    select.appendChild(placeholder);

    (vocab[key] || []).forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });

    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = TRAIT_HINTS[key];

    label.appendChild(select);
    label.appendChild(hint);
    container.appendChild(label);
  });
}

function buildTestInputs() {
  const container = $("test-traits");
  container.innerHTML = "";

  TRAIT_KEYS.forEach((key) => {
    const wrapper = document.createElement("div");

    const traitLabel = document.createElement("label");
    traitLabel.appendChild(document.createTextNode(TRAIT_LABELS[key]));
    const select = document.createElement("select");
    select.id = `test-trait-${key}`;
    (vocab[key] || []).forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
    traitLabel.appendChild(select);

    const probLabel = document.createElement("label");
    probLabel.appendChild(document.createTextNode("ความมั่นใจ (0–1)"));
    const prob = document.createElement("input");
    prob.type = "number";
    prob.id = `test-prob-${key}`;
    prob.step = "0.05";
    prob.min = "0";
    prob.max = "1";
    prob.value = "0.9";
    probLabel.appendChild(prob);

    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = "โมเดลจริงส่งมาเป็น % ระบบหารร้อยให้ก่อนคิดคะแนน";
    probLabel.appendChild(hint);

    wrapper.appendChild(traitLabel);
    wrapper.appendChild(probLabel);
    container.appendChild(wrapper);
  });
}

// ── Rules table ───────────────────────────────────────────
function setRulesLoading(isLoading) {
  $("rules-loading").classList.toggle("hidden", !isLoading);
  $("rules-table-wrap").classList.toggle("loading", isLoading);
  if (isLoading) $("rules-empty").classList.add("hidden");
}

function renderRules(groups) {
  const tbody = $("rules-tbody");
  tbody.innerHTML = "";
  $("rules-empty").classList.toggle("hidden", groups.length > 0);
  $("rules-count").textContent = groups.length ? `${groups.length} กลุ่ม` : "";

  groups.forEach((group) => {
    const row = document.createElement("tr");

    const cells = [
      group.code,
      group.name,
      group.shape,
      group.apex,
      group.base,
      group.margin,
    ];
    cells.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 1) cell.classList.add("wrap");
      row.appendChild(cell);
    });

    const activeCell = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = group.is_active ? "pill" : "pill off";
    pill.textContent = group.is_active ? "ใช้งาน" : "ปิด";
    activeCell.appendChild(pill);
    row.appendChild(activeCell);

    const actions = document.createElement("td");
    const editBtn = document.createElement("button");
    editBtn.className = "link";
    editBtn.textContent = "แก้ไข";
    editBtn.addEventListener("click", () => openEditor(group));

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "link danger";
    deleteBtn.textContent = "ลบ";
    deleteBtn.addEventListener("click", () => deleteRule(group, deleteBtn));

    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(actions);

    tbody.appendChild(row);
  });
}

async function loadRules() {
  setRulesLoading(true);
  try {
    const data = await api("/api/rules");
    renderRules(data.groups || []);
  } catch (error) {
    const message = describeError(error);
    if (message) flash(message, true);
  } finally {
    setRulesLoading(false);
  }
}

async function deleteRule(group, button) {
  if (!confirm(`ลบกลุ่ม ${group.code} (${group.name}) ?`)) return;
  setBusy(button, "กำลังลบ…");
  try {
    await api(`/api/rules/${group.id}`, { method: "DELETE" });
    flash(`ลบ ${group.code} แล้ว`);
    await loadRules();
  } catch (error) {
    clearBusy(button);
    const message = describeError(error);
    if (message) flash(message, true);
  }
  // On success the row is replaced by loadRules(), so the button is gone.
}

// ── Editor ────────────────────────────────────────────────
function openEditor(group) {
  $("editor-error").classList.add("hidden");
  $("editor-title").textContent = group ? `แก้ไข ${group.code}` : "เพิ่มกลุ่มใหม่";
  $("rule-id").value = group ? group.id : "";
  $("rule-code").value = group ? group.code : "";
  $("rule-name").value = group ? group.name : "";
  $("rule-active").checked = group ? group.is_active : true;

  TRAIT_KEYS.forEach((key) => {
    $(`rule-trait-${key}`).value = group ? group[key] : "";
  });

  $("editor-backdrop").classList.remove("hidden");
}

function closeEditor() {
  $("editor-backdrop").classList.add("hidden");
}

$("new-rule-btn").addEventListener("click", () => openEditor(null));
$("editor-cancel").addEventListener("click", closeEditor);
$("editor-backdrop").addEventListener("click", (event) => {
  if (event.target === $("editor-backdrop")) closeEditor();
});

$("rule-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorEl = $("editor-error");
  const button = $("editor-submit");
  errorEl.classList.add("hidden");

  const payload = {
    code: $("rule-code").value.trim(),
    name: $("rule-name").value.trim(),
    is_active: $("rule-active").checked,
  };
  TRAIT_KEYS.forEach((key) => {
    payload[key] = $(`rule-trait-${key}`).value;
  });

  const id = $("rule-id").value;
  setBusy(button, "กำลังบันทึก…");
  try {
    await api(id ? `/api/rules/${id}` : "/api/rules", {
      method: id ? "PUT" : "POST",
      body: JSON.stringify(payload),
    });
    closeEditor();
    flash(id ? `บันทึก ${payload.code} แล้ว` : `เพิ่ม ${payload.code} แล้ว`);
    await loadRules();
  } catch (error) {
    const message = describeError(error);
    if (message) {
      errorEl.textContent = message;
      errorEl.classList.remove("hidden");
    }
  } finally {
    clearBusy(button);
  }
});

// ── Test bench ────────────────────────────────────────────
$("run-test-btn").addEventListener("click", async () => {
  const button = $("run-test-btn");
  const traits = {};
  const probs = {};
  TRAIT_KEYS.forEach((key) => {
    traits[key] = $(`test-trait-${key}`).value;
    probs[key] = parseFloat($(`test-prob-${key}`).value);
  });

  const payload = {
    traits,
    probs,
    conf_th: parseFloat($("test-conf-th").value),
    top_n: 3,
  };

  setBusy(button, "กำลังคำนวณ…");
  try {
    const data = await api("/api/rules/test", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderTestResults(data.results || []);
  } catch (error) {
    const message = describeError(error);
    if (message) flash(message, true);
  } finally {
    clearBusy(button);
  }
});

function renderTestResults(results) {
  const table = $("test-table");
  const tbody = $("test-tbody");
  tbody.innerHTML = "";

  if (results.length === 0) {
    table.classList.add("hidden");
    flash("ยังไม่มีกฎที่เปิดใช้งาน — เพิ่มกลุ่มก่อน", true);
    return;
  }

  results.forEach((result, index) => {
    const row = document.createElement("tr");
    [index + 1, result.code, result.name, `${result.matched} / 4`, result.score]
      .forEach((value, cellIndex) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (cellIndex === 2) cell.classList.add("wrap");
        row.appendChild(cell);
      });
    tbody.appendChild(row);
  });

  table.classList.remove("hidden");
}

// ── Boot ──────────────────────────────────────────────────
async function boot() {
  const data = await api("/api/rules/vocab");
  vocab = data.vocab || {};
  buildTraitSelectors();
  buildTestInputs();
  await loadRules();
}

(async () => {
  try {
    const me = await api("/api/admin/me");
    showApp(me.username);
    await boot();
  } catch (error) {
    if (!(error instanceof Unauthorized)) {
      showLogin();
    }
  }
})();
