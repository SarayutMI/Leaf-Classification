# src/rules/router.py
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.auth.dependencies import require_session
from src.rules import repository, service
from src.rules.schemas import (
    TRAIT_KEYS,
    VOCAB,
    RuleGroupIn,
    RuleTestRequest,
    VarietyBatchIn,
    VarietyIn,
)

# The session check lives on the router, not on each route: a new endpoint
# added here is protected by default instead of only when someone remembers
# the Depends().
router = APIRouter(
    prefix="/api/rules",
    tags=["rules"],
    dependencies=[Depends(require_session)],
)
logger = logging.getLogger(__name__)


# Everything behind require_session is per-user data. Without this a browser
# is free to reuse a 200 it fetched while logged in — served straight from its
# own cache, never reaching the server, so the session check cannot run.
NO_STORE = {"Cache-Control": "no-store, private", "Pragma": "no-cache"}


def _ok(data, code: int = 200):
    return JSONResponse(
        status_code=code,
        content={"code": code, "status": "success", "data": data},
        headers=NO_STORE,
    )


def _field_errors(errors: list[dict], message: str = "Validation failed"):
    """422 that points at the exact row and field the caller got wrong.

    The add form submits many names at once, so "a name is duplicated" is not
    actionable — each entry carries the `index` of the offending item (null for
    form-level fields such as group_id) and the `field` name.
    """
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "status": "error",
            "message": message,
            "errors": {"type": "VALIDATION_ERROR", "fields": errors},
        },
        headers=NO_STORE,
    )


def _combination_text(row: dict) -> str:
    combination = " / ".join(row[key] for key in TRAIT_KEYS)
    return f"Another group already uses the combination {combination}"


def _error(code: int, message: str, details: str | None = None):
    content = {"code": code, "status": "error", "message": message}
    if details:
        content["errors"] = {"type": "VALIDATION_ERROR", "details": details}
    return JSONResponse(status_code=code, content=content, headers=NO_STORE)


# Static paths are declared before /{group_id} so they are never swallowed by
# the parameterised route.
@router.get("/vocab")
async def get_vocab():
    """The class list per trait, so the page builds its dropdowns from the
    same source the models were configured with."""
    return _ok({"vocab": VOCAB})


@router.post("/test")
async def test_rules(body: RuleTestRequest):
    try:
        results = service.match_group(
            traits=body.traits,
            probs=body.probs,
            conf_th=body.conf_th,
            top_n=body.top_n,
        )
    except Exception:
        logger.exception("Rule test failed")
        return _error(500, "Database error")
    return _ok({"results": results})


@router.get("")
async def list_rules():
    try:
        groups = repository.list_groups()
        counts = repository.variety_counts_by_group()
        for group in groups:
            group["variety_count"] = counts.get(group["id"], 0)
    except Exception:
        logger.exception("Failed to list rule groups")
        return _error(500, "Database error")
    return _ok({"groups": groups})


@router.post("")
async def create_rule(body: RuleGroupIn):
    try:
        row = body.to_row()
        if repository.code_exists(body.code):
            return _error(409, "Duplicate code", f"Rule code '{body.code}' already exists")
        if repository.combination_exists(row):
            return _error(409, "Duplicate rule", _combination_text(row))
        group_id = repository.create_group(row)
        service.invalidate_cache()
        group = repository.get_group(group_id)
    except Exception:
        logger.exception("Failed to create rule group")
        return _error(500, "Database error")
    return _ok({"group": group}, code=201)


# ── Varieties ────────────────────────────────────────────────
# Declared before /{group_id} for the same reason as /vocab and /test.


@router.get("/varieties")
async def list_varieties():
    try:
        varieties = repository.list_varieties()
    except Exception:
        logger.exception("Failed to list varieties")
        return _error(500, "Database error")
    return _ok({"varieties": varieties})


def _validate_variety_names(items, exclude_id=None) -> list[dict]:
    """Empty, duplicated-in-batch and already-taken names, reported by index.

    Comparison is case-insensitive, matching the database lookup — "มันเสือ" and
    "มันเสือ " are the same name to a person, and should be to the form too.
    """
    errors = []
    seen: dict[str, int] = {}

    for index, item in enumerate(items):
        if not item.name:
            errors.append({"index": index, "field": "name",
                           "message": "กรุณากรอกชื่อชนิดมัน"})
            continue
        key = item.name.lower()
        if key in seen:
            errors.append({"index": index, "field": "name",
                           "message": f"ชื่อซ้ำกับบรรทัดที่ {seen[key] + 1}"})
            continue
        seen[key] = index

    if errors:
        # Skip the database round trip while there is still a local problem.
        return errors

    taken = repository.existing_variety_names(list(seen), exclude_id=exclude_id)
    for key, index in seen.items():
        if key in taken:
            errors.append({"index": index, "field": "name",
                           "message": "ชื่อนี้มีอยู่แล้วในระบบ"})

    errors.sort(key=lambda e: e["index"])
    return errors


@router.post("/varieties")
async def create_varieties(body: VarietyBatchIn):
    """Add several varieties to one group in a single submission."""
    try:
        errors = []
        if not repository.group_ids_that_exist([body.group_id]):
            errors.append({"index": None, "field": "group_id",
                           "message": "ไม่พบกลุ่มนี้"})
        errors += _validate_variety_names(body.items)
        if errors:
            return _field_errors(errors)

        ids = repository.create_varieties(
            body.group_id, [item.model_dump() for item in body.items]
        )
        varieties = repository.get_varieties(ids)
    except Exception:
        logger.exception("Failed to create varieties")
        return _error(500, "Database error")
    return _ok({"varieties": varieties, "created": len(varieties)}, code=201)


@router.get("/varieties/{variety_id}")
async def get_variety(variety_id: int):
    try:
        variety = repository.get_variety(variety_id)
    except Exception:
        logger.exception("Failed to read variety %s", variety_id)
        return _error(500, "Database error")
    if variety is None:
        return _error(404, "Variety not found")
    return _ok({"variety": variety})


@router.put("/varieties/{variety_id}")
async def update_variety(variety_id: int, body: VarietyIn):
    try:
        errors = []
        if not repository.group_ids_that_exist([body.group_id]):
            errors.append({"index": None, "field": "group_id",
                           "message": "ไม่พบกลุ่มนี้"})
        errors += _validate_variety_names([body], exclude_id=variety_id)
        if errors:
            return _field_errors(errors)

        if not repository.update_variety(variety_id, body.to_row()):
            return _error(404, "Variety not found")
        variety = repository.get_variety(variety_id)
    except Exception:
        logger.exception("Failed to update variety %s", variety_id)
        return _error(500, "Database error")
    return _ok({"variety": variety})


@router.delete("/varieties/{variety_id}")
async def delete_variety(variety_id: int):
    try:
        deleted = repository.delete_variety(variety_id)
    except Exception:
        logger.exception("Failed to delete variety %s", variety_id)
        return _error(500, "Database error")
    if not deleted:
        return _error(404, "Variety not found")
    return _ok({"deleted": variety_id})


@router.get("/{group_id}")
async def get_rule(group_id: int):
    try:
        group = repository.get_group(group_id)
    except Exception:
        logger.exception("Failed to read rule group %s", group_id)
        return _error(500, "Database error")
    if group is None:
        return _error(404, "Rule group not found")
    return _ok({"group": group})


@router.put("/{group_id}")
async def update_rule(
    group_id: int,
    body: RuleGroupIn,
):
    try:
        row = body.to_row()
        if repository.code_exists(body.code, exclude_id=group_id):
            return _error(409, "Duplicate code", f"Rule code '{body.code}' already exists")
        if repository.combination_exists(row, exclude_id=group_id):
            return _error(409, "Duplicate rule", _combination_text(row))
        if not repository.update_group(group_id, row):
            return _error(404, "Rule group not found")
        service.invalidate_cache()
        group = repository.get_group(group_id)
    except Exception:
        logger.exception("Failed to update rule group %s", group_id)
        return _error(500, "Database error")
    return _ok({"group": group})


@router.delete("/{group_id}")
async def delete_rule(group_id: int):
    try:
        deleted = repository.delete_group(group_id)
    except Exception:
        logger.exception("Failed to delete rule group %s", group_id)
        return _error(500, "Database error")
    if not deleted:
        return _error(404, "Rule group not found")
    service.invalidate_cache()
    return _ok({"deleted": group_id})
