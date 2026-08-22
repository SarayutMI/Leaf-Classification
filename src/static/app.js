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
// Both lists are small — at most 128 groups — so filtering happens here rather
// than as a query per keystroke. Keep the full sets; render the filtered view.
let groups = [];
let varieties = [];

const $ = (id) => document.getElementById(id);

// ── Token ─────────────────────────────────────────────────
// The API is Bearer-only so other services can call it too, which means the
// page has to hold the token itself. localStorage is readable by any script
// on this origin — keep third-party scripts out of this page.
const TOKEN_KEY = "leaf_admin_token";

function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;   // private mode with site data blocked
  }
}

function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* the session still works for this page load */
  }
}

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
  const headers = { ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(path, { ...options, headers });

  if (response.status === 401) {
    setToken(null);   // expired or revoked — do not keep retrying with it
    showLogin();
    throw new Unauthorized("Not authenticated");
  }

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const errors = body.errors || {};
    const failure = new Error(
      errors.details || body.message || `Request failed (${response.status})`
    );
    // Per-row validation messages, placed on the offending input by the caller.
    if (errors.fields) failure.fields = errors.fields;
    throw failure;
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
    setToken(data.access_token);
    showApp(data.user && data.user.username);
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
    await fetch("/api/admin/logout", { method: "POST" });
  } catch {
    /* the token is discarded either way */
  } finally {
    setToken(null);
    clearBusy(button);
    showLogin();
  }
});

// ── Tabs ──────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    ["rules", "varieties", "test"].forEach((name) => {
      $(`tab-${name}`).classList.toggle("hidden", tab.dataset.tab !== name);
    });
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
  if (isLoading) {
    $("rules-empty").classList.add("hidden");
    $("rules-no-match").classList.add("hidden");
  }
}

$("rules-search").addEventListener("input", renderRules);
$("rules-clear").addEventListener("click", () => {
  $("rules-search").value = "";
  renderRules();
  $("rules-search").focus();
});

function matchesRuleSearch(group, term) {
  if (!term) return true;
  return [group.code, group.name, group.shape, group.apex, group.base, group.margin]
    .some((value) => String(value).toLowerCase().includes(term));
}

function renderRules() {
  const term = $("rules-search").value.trim().toLowerCase();
  const visible = groups.filter((group) => matchesRuleSearch(group, term));

  $("rules-clear").classList.toggle("hidden", !term);
  $("rules-empty").classList.toggle("hidden", groups.length > 0);
  $("rules-no-match").classList.toggle(
    "hidden", groups.length === 0 || visible.length > 0
  );
  $("rules-count").textContent = groups.length
    ? (term ? `${visible.length} / ${groups.length} กลุ่ม` : `${groups.length} กลุ่ม`)
    : "";

  const tbody = $("rules-tbody");
  tbody.innerHTML = "";

  visible.forEach((group) => {
    const row = document.createElement("tr");

    const cells = [
      group.code,
      group.name,
      group.shape,
      group.apex,
      group.base,
      group.margin,
      group.variety_count ?? 0,
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
    groups = data.groups || [];
  } catch (error) {
    const message = describeError(error);
    if (message) flash(message, true);
  } finally {
    // Render either way: a failed load must still leave a usable "ทุกกลุ่ม"
    // filter rather than an empty <select>.
    buildGroupFilter();
    renderRules();
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
    await loadVarieties();   // the group's varieties were cascade-deleted
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
    if (id) await loadVarieties();   // the group label they display changed
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


// ── Varieties ─────────────────────────────────────────────
function setVarietiesLoading(isLoading) {
  $("varieties-loading").classList.toggle("hidden", !isLoading);
  $("varieties-table-wrap").classList.toggle("loading", isLoading);
  if (isLoading) {
    $("varieties-empty").classList.add("hidden");
    $("varieties-no-match").classList.add("hidden");
  }
}

/** Rebuild the group filter, keeping the current choice if it still exists. */
function buildGroupFilter() {
  const select = $("varieties-group-filter");
  const previous = select.value;
  select.innerHTML = "";

  const all = document.createElement("option");
  all.value = "";
  all.textContent = "ทุกกลุ่ม";
  select.appendChild(all);

  groups.forEach((group) => {
    const option = document.createElement("option");
    option.value = String(group.id);
    option.textContent = `${group.code} — ${group.name}`;
    select.appendChild(option);
  });

  select.value = groups.some((g) => String(g.id) === previous) ? previous : "";
}

$("varieties-search").addEventListener("input", renderVarieties);
$("varieties-group-filter").addEventListener("change", renderVarieties);
$("varieties-clear").addEventListener("click", () => {
  $("varieties-search").value = "";
  $("varieties-group-filter").value = "";
  renderVarieties();
  $("varieties-search").focus();
});

function groupLabel(groupId) {
  const group = groups.find((g) => g.id === groupId);
  return group ? `${group.code} — ${group.name}` : `#${groupId}`;
}

function renderVarieties() {
  const term = $("varieties-search").value.trim().toLowerCase();
  const groupFilter = $("varieties-group-filter").value;

  const visible = varieties.filter((variety) => {
    if (groupFilter && String(variety.group_id) !== groupFilter) return false;
    // Searching the group label too, so "G1" finds everything in that group.
    if (!term) return true;
    return `${variety.name} ${groupLabel(variety.group_id)}`
      .toLowerCase()
      .includes(term);
  });

  const filtering = Boolean(term || groupFilter);
  $("varieties-clear").classList.toggle("hidden", !filtering);
  $("varieties-empty").classList.toggle("hidden", varieties.length > 0);
  $("varieties-no-match").classList.toggle(
    "hidden", varieties.length === 0 || visible.length > 0
  );
  $("varieties-count").textContent = varieties.length
    ? (filtering
        ? `${visible.length} / ${varieties.length} ชนิด`
        : `${varieties.length} ชนิด`)
    : "";

  const tbody = $("varieties-tbody");
  tbody.innerHTML = "";

  visible.forEach((variety) => {
    const row = document.createElement("tr");

    const nameCell = document.createElement("td");
    nameCell.textContent = variety.name;
    nameCell.classList.add("wrap");
    row.appendChild(nameCell);

    const groupCell = document.createElement("td");
    groupCell.textContent = groupLabel(variety.group_id);
    groupCell.classList.add("wrap");
    row.appendChild(groupCell);

    const activeCell = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = variety.is_active ? "pill" : "pill off";
    pill.textContent = variety.is_active ? "ใช้งาน" : "ปิด";
    activeCell.appendChild(pill);
    row.appendChild(activeCell);

    const actions = document.createElement("td");
    const editBtn = document.createElement("button");
    editBtn.className = "link";
    editBtn.textContent = "แก้ไข";
    editBtn.addEventListener("click", () => openVarietyEditor(variety));

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "link danger";
    deleteBtn.textContent = "ลบ";
    deleteBtn.addEventListener("click", () => deleteVariety(variety, deleteBtn));

    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(actions);

    tbody.appendChild(row);
  });
}

async function loadVarieties() {
  setVarietiesLoading(true);
  try {
    const data = await api("/api/rules/varieties");
    varieties = data.varieties || [];
    renderVarieties();
  } catch (error) {
    const message = describeError(error);
    if (message) flash(message, true);
  } finally {
    setVarietiesLoading(false);
  }
}

async function deleteVariety(variety, button) {
  if (!confirm(`ลบชนิดมัน "${variety.name}" ?`)) return;
  setBusy(button, "กำลังลบ…");
  try {
    await api(`/api/rules/varieties/${variety.id}`, { method: "DELETE" });
    flash(`ลบ ${variety.name} แล้ว`);
    await loadVarieties();
    await loadRules();   // the group variety counts changed
  } catch (error) {
    clearBusy(button);
    const message = describeError(error);
    if (message) flash(message, true);
  }
}

// ── Variety editor ────────────────────────────────────────
function buildGroupSelect(selectedId) {
  const select = $("variety-group");
  select.innerHTML = "";

  if (groups.length === 0) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "— ยังไม่มีกลุ่มการจำแนก —";
    select.appendChild(empty);
    select.disabled = true;
    return;
  }

  select.disabled = false;
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— เลือกกลุ่ม —";
  select.appendChild(placeholder);

  groups.forEach((group) => {
    const option = document.createElement("option");
    option.value = String(group.id);
    option.textContent = `${group.code} — ${group.name}`;
    select.appendChild(option);
  });

  select.value = selectedId ? String(selectedId) : "";
}

function addNameRow(value = "", isActive = true) {
  const container = $("variety-names");

  const row = document.createElement("div");
  row.className = "name-row";

  const inputs = document.createElement("div");
  inputs.className = "name-row-inputs";

  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "มันเสือ";
  input.maxLength = 255;
  input.value = value;
  input.dataset.nameInput = "1";

  const activeLabel = document.createElement("label");
  activeLabel.className = "check";
  const activeBox = document.createElement("input");
  activeBox.type = "checkbox";
  activeBox.checked = isActive;
  activeBox.dataset.activeInput = "1";
  activeLabel.appendChild(activeBox);
  activeLabel.appendChild(document.createTextNode(" ใช้งาน"));

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "row-remove";
  remove.textContent = "ลบ";
  remove.addEventListener("click", () => {
    // Never leave the form with nothing to type in.
    if (container.querySelectorAll(".name-row").length > 1) row.remove();
    else input.value = "";
  });

  const error = document.createElement("span");
  error.className = "field-error hidden";

  inputs.appendChild(input);
  inputs.appendChild(activeLabel);
  inputs.appendChild(remove);
  row.appendChild(inputs);
  row.appendChild(error);
  container.appendChild(row);

  // Typing clears the server-side complaint about this row.
  input.addEventListener("input", () => {
    input.classList.remove("invalid");
    error.classList.add("hidden");
  });

  return input;
}

function clearFieldErrors() {
  $("variety-error").classList.add("hidden");
  document.querySelectorAll("#variety-form .field-error").forEach((el) => {
    el.textContent = "";
    el.classList.add("hidden");
  });
  document.querySelectorAll("#variety-form .invalid").forEach((el) => {
    el.classList.remove("invalid");
  });
}

/** Paint a 422 from the API onto the row it belongs to. */
function showFieldErrors(fields) {
  const rows = document.querySelectorAll("#variety-names .name-row");
  const unplaced = [];

  fields.forEach((field) => {
    if (field.index === null || field.index === undefined) {
      const target = document.querySelector(
        `#variety-form .field-error[data-error-for="${field.field}"]`
      );
      if (target) {
        target.textContent = field.message;
        target.classList.remove("hidden");
        $("variety-group").classList.add("invalid");
        return;
      }
      unplaced.push(field.message);
      return;
    }

    const row = rows[field.index];
    if (!row) {
      unplaced.push(`บรรทัดที่ ${field.index + 1}: ${field.message}`);
      return;
    }
    row.querySelector("input[data-name-input]").classList.add("invalid");
    const error = row.querySelector(".field-error");
    error.textContent = field.message;
    error.classList.remove("hidden");
  });

  if (unplaced.length) {
    const box = $("variety-error");
    box.textContent = unplaced.join(" · ");
    box.classList.remove("hidden");
  }
}

function openVarietyEditor(variety) {
  clearFieldErrors();
  $("variety-title").textContent = variety ? `แก้ไข ${variety.name}` : "เพิ่มชนิดมัน";
  $("variety-id").value = variety ? variety.id : "";
  buildGroupSelect(variety ? variety.group_id : null);

  // Editing touches one row; adding starts with one and grows.
  $("variety-names").innerHTML = "";
  addNameRow(variety ? variety.name : "", variety ? variety.is_active : true);

  const adding = !variety;
  $("add-name-row").classList.toggle("hidden", !adding);
  $("variety-names-note").classList.toggle("hidden", !adding);
  document.querySelectorAll("#variety-names .row-remove").forEach((b) => {
    b.classList.toggle("hidden", !adding);
  });

  $("variety-backdrop").classList.remove("hidden");
}

function closeVarietyEditor() {
  $("variety-backdrop").classList.add("hidden");
}

$("new-variety-btn").addEventListener("click", () => openVarietyEditor(null));
$("variety-cancel").addEventListener("click", closeVarietyEditor);
$("add-name-row").addEventListener("click", () => addNameRow().focus());
$("variety-backdrop").addEventListener("click", (event) => {
  if (event.target === $("variety-backdrop")) closeVarietyEditor();
});

function collectNameRows() {
  return Array.from(document.querySelectorAll("#variety-names .name-row")).map((row) => ({
    name: row.querySelector("input[data-name-input]").value.trim(),
    is_active: row.querySelector("input[data-active-input]").checked,
  }));
}

$("variety-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearFieldErrors();

  const button = $("variety-submit");
  const id = $("variety-id").value;
  const groupValue = $("variety-group").value;

  if (!groupValue) {
    showFieldErrors([{ index: null, field: "group_id", message: "กรุณาเลือกกลุ่ม" }]);
    return;
  }

  const items = collectNameRows();
  const groupId = parseInt(groupValue, 10);
  const payload = id
    ? { group_id: groupId, ...items[0] }
    : { group_id: groupId, items };

  setBusy(button, "กำลังบันทึก…");
  try {
    const data = await api(id ? `/api/rules/varieties/${id}` : "/api/rules/varieties", {
      method: id ? "PUT" : "POST",
      body: JSON.stringify(payload),
    });
    closeVarietyEditor();
    flash(id ? `บันทึก ${items[0].name} แล้ว` : `เพิ่ม ${data.created} ชนิดแล้ว`);
    await loadVarieties();
    await loadRules();   // the group variety counts changed
  } catch (error) {
    if (error instanceof Unauthorized) return;
    if (error.fields) {
      showFieldErrors(error.fields);
      return;
    }
    const box = $("variety-error");
    box.textContent = describeError(error);
    box.classList.remove("hidden");
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
  await loadRules();        // fills `groups`, which the variety editor needs
  await loadVarieties();
}

(async () => {
  // No stored token means no session — skip the round trip.
  if (!getToken()) {
    showLogin();
    return;
  }
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
