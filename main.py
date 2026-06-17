from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import sqlite3
import secrets
from datetime import datetime, timezone
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

ADMIN_KEY = os.environ.get("ADMIN_KEY")
DB_PATH = "clients.db"

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "60 per minute"],
    storage_uri="memory://",
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                api_key     TEXT    NOT NULL UNIQUE,
                active      INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT    NOT NULL,
                last_used   TEXT,
                request_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()

    # Migrate: if the old single API_KEY env var exists and is not yet in DB,
    # seed it as "Client 1" so existing Bubble connections keep working.
    legacy_key = os.environ.get("API_KEY")
    if legacy_key:
        with get_db() as conn:
            exists = conn.execute(
                "SELECT id FROM clients WHERE api_key = ?", (legacy_key,)
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO clients (name, api_key, active, created_at) VALUES (?, ?, 1, ?)",
                    ("Client 1 (legacy)", legacy_key, _now())
                )
                conn.commit()


def _now():
    return datetime.now(timezone.utc).isoformat()


init_db()


# ---------------------------------------------------------------------------
# Auth decorators
# ---------------------------------------------------------------------------

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key:
            return jsonify({"error": "Unauthorized — missing API key"}), 401
        with get_db() as conn:
            row = conn.execute(
                "SELECT id, active FROM clients WHERE api_key = ?", (key,)
            ).fetchone()
            if not row:
                return jsonify({"error": "Unauthorized — invalid API key"}), 401
            if not row["active"]:
                return jsonify({"error": "Unauthorized — this API key has been revoked"}), 401
            conn.execute(
                "UPDATE clients SET last_used = ?, request_count = request_count + 1 WHERE id = ?",
                (_now(), row["id"])
            )
            conn.commit()
        return f(*args, **kwargs)
    return decorated


def require_admin_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-Admin-Key")
        if not key or key != ADMIN_KEY:
            return jsonify({"error": "Unauthorized — invalid or missing admin key"}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Maths
# ---------------------------------------------------------------------------

def get_overall_score(profit_probability, margin_health):
    return round((profit_probability * 0.6) + (margin_health * 0.4), 2)


def get_margin_health(net_sales, cogs):
    net_sales = float(net_sales)
    cogs = float(cogs)
    if net_sales == 0:
        return 0.0
    gross_margin = ((net_sales - cogs) / net_sales) * 100
    return round(max(0.0, min(100.0, gross_margin)), 2)


def get_recommendation(prob):
    if prob < 40:
        return "High return rates or discounts are eroding profit — review pricing and return policies immediately."
    elif prob < 70:
        return "Moderate risk detected — consider reducing discounts and monitoring return trends closely."
    else:
        return "Strong performer — maintain current pricing and sales strategy."


def bayesian_update(prior, likelihood_if_profitable, likelihood_if_not_profitable):
    numerator = likelihood_if_profitable * prior
    denominator = numerator + likelihood_if_not_profitable * (1 - prior)
    if denominator == 0:
        return prior
    return numerator / denominator


def calculate_profit_probability(net_sales, cogs, profit, returns, discounts):
    net_sales = float(net_sales)
    cogs = float(cogs)
    profit = float(profit)
    returns = float(returns)
    discounts = float(discounts)

    posterior = 0.50

    gross_margin = ((net_sales - cogs) / net_sales) * 100 if net_sales > 0 else 0
    if gross_margin >= 40:
        posterior = bayesian_update(posterior, 0.82, 0.22)
    elif gross_margin >= 20:
        posterior = bayesian_update(posterior, 0.60, 0.45)
    else:
        posterior = bayesian_update(posterior, 0.28, 0.72)

    return_rate_pct = (returns / net_sales) * 100 if net_sales > 0 else 0
    if return_rate_pct <= 5:
        posterior = bayesian_update(posterior, 0.78, 0.28)
    elif return_rate_pct <= 15:
        posterior = bayesian_update(posterior, 0.55, 0.50)
    else:
        posterior = bayesian_update(posterior, 0.22, 0.78)

    discount_rate_pct = (discounts / net_sales) * 100 if net_sales > 0 else 0
    if discount_rate_pct <= 5:
        posterior = bayesian_update(posterior, 0.72, 0.32)
    elif discount_rate_pct <= 15:
        posterior = bayesian_update(posterior, 0.55, 0.50)
    else:
        posterior = bayesian_update(posterior, 0.28, 0.72)

    profit_margin_pct = (profit / net_sales) * 100 if net_sales > 0 else 0
    if profit_margin_pct >= 20:
        posterior = bayesian_update(posterior, 0.88, 0.12)
    elif profit_margin_pct >= 5:
        posterior = bayesian_update(posterior, 0.65, 0.38)
    elif profit_margin_pct >= 0:
        posterior = bayesian_update(posterior, 0.42, 0.58)
    else:
        posterior = bayesian_update(posterior, 0.12, 0.88)

    return_rate_decimal = returns / (net_sales + 1)
    adjusted_profit = profit * (1 - return_rate_decimal) - discounts
    uncertainty = (abs(profit) * 0.10) + (returns * 0.30) + (discounts * 0.20)

    simulations = np.random.normal(adjusted_profit, uncertainty + 1, 1000)
    mc_probability = np.mean(simulations >= profit * 0.9)

    combined = (posterior * 0.60) + (mc_probability * 0.40)

    # Confidence interval from simulation distribution (5th–95th percentile)
    sim_probabilities = (simulations >= profit * 0.9).astype(float)
    sim_chunks = sim_probabilities.reshape(10, 100)
    chunk_means = sim_chunks.mean(axis=1)
    blended_chunks = chunk_means * 0.40 + posterior * 0.60
    ci_low = round(float(np.percentile(blended_chunks, 5)) * 100, 2)
    ci_high = round(float(np.percentile(blended_chunks, 95)) * 100, 2)

    return round(combined * 100, 2), ci_low, ci_high


def calculate_breakeven_units(net_sales, cogs, returns, discounts, units_sold):
    net_sales = float(net_sales)
    cogs = float(cogs)
    returns = float(returns)
    discounts = float(discounts)
    units_sold = float(units_sold)

    if units_sold <= 0 or net_sales <= 0:
        return None

    revenue_per_unit = net_sales / units_sold
    cogs_per_unit = cogs / units_sold
    returns_per_unit = returns / units_sold
    discounts_per_unit = discounts / units_sold

    net_revenue_per_unit = revenue_per_unit - returns_per_unit - discounts_per_unit
    contribution_per_unit = net_revenue_per_unit - cogs_per_unit

    if contribution_per_unit <= 0:
        return None

    breakeven_units = cogs / contribution_per_unit
    return round(breakeven_units, 0)


def calculate_cogs_stress(net_sales, cogs, profit, returns, discounts, cogs_increase_pct):
    stressed_cogs = cogs * (1 + float(cogs_increase_pct) / 100)
    stressed_profit = profit - (stressed_cogs - cogs)
    stressed_prob, _, _ = calculate_profit_probability(
        net_sales, stressed_cogs, stressed_profit, returns, discounts
    )
    return round(stressed_prob, 2)


def calculate_optimal_price(net_sales, cogs, returns, discounts, current_price, target_margin):
    net_sales = float(net_sales)
    cogs = float(cogs)
    returns = float(returns)
    discounts = float(discounts)
    current_price = float(current_price)
    target_margin = float(target_margin)

    if net_sales <= 0 or current_price <= 0:
        return None

    return_rate = min(returns / net_sales, 0.99)
    discount_rate = min(discounts / net_sales, 0.99)
    cogs_per_unit = cogs * current_price / net_sales

    effective_denominator = (1 - return_rate) * (1 - discount_rate)
    if effective_denominator <= 0:
        return None

    breakeven_price = cogs_per_unit / effective_denominator

    target = max(0.0, min(target_margin, 99.0))
    margin_denominator = (1 - target / 100) * effective_denominator
    if margin_denominator <= 0:
        return None

    recommended_price = cogs_per_unit / margin_denominator

    price_gap = recommended_price - current_price
    current_price_is_healthy = price_gap <= 0.01
    if price_gap > 0.01:
        pricing_action = f"Price is below the {target_margin}% margin target. Raise to at least {recommended_price:.2f} to hit your target."
    elif price_gap < -0.01:
        pricing_action = f"Your current price already exceeds the {target_margin}% target. Minimum price to maintain target: {recommended_price:.2f}."
    else:
        pricing_action = f"Price is exactly at the {target_margin}% margin target."

    return {
        "current_price": round(current_price, 2),
        "breakeven_price": round(breakeven_price, 2),
        "minimum_price_for_target": round(recommended_price, 2),
        "current_price_is_healthy": current_price_is_healthy,
        "target_margin_pct": round(target_margin, 1),
        "pricing_action": pricing_action,
    }


DEFAULT_TARGET_MARGIN = 40.0


def build_result(item, target_margin=None):
    net_sales = item.get("net_sales", 0)
    cogs = item.get("cogs", 0)
    profit = item.get("profit", 0)
    returns = item.get("returns", 0)
    discounts = item.get("discounts", 0)

    prob, ci_low, ci_high = calculate_profit_probability(
        net_sales, cogs, profit, returns, discounts
    )
    margin = get_margin_health(net_sales, cogs)
    result = {
        "product": item.get("product", "Unknown"),
        "profit_probability": prob,
        "probability_range": f"{ci_low}%–{ci_high}%",
        "probability_low": ci_low,
        "probability_high": ci_high,
        "requires_attention": bool(prob < 40),
        "risk_level": "High" if prob < 40 else "Medium" if prob < 70 else "Low",
        "recommendation": get_recommendation(prob),
        "margin_health": margin,
        "overall_score": get_overall_score(prob, margin),
    }

    # Break-even units (optional — requires units_sold in input)
    units_sold = item.get("units_sold")
    if units_sold is not None:
        beu = calculate_breakeven_units(net_sales, cogs, returns, discounts, units_sold)
        result["breakeven_units"] = beu
        if beu is not None:
            result["breakeven_units_note"] = (
                f"You need to sell at least {int(beu)} units to break even."
            )

    # COGS stress-test (optional — requires cogs_increase_pct in input)
    cogs_increase_pct = item.get("cogs_increase_pct")
    if cogs_increase_pct is not None:
        stressed_prob = calculate_cogs_stress(
            net_sales, cogs, profit, returns, discounts, cogs_increase_pct
        )
        result["cogs_stress"] = {
            "cogs_increase_pct": float(cogs_increase_pct),
            "stressed_profit_probability": stressed_prob,
            "impact": round(stressed_prob - prob, 2),
            "stress_note": (
                f"If COGS rise {cogs_increase_pct}%, profit probability drops from "
                f"{prob}% to {stressed_prob}% ({round(stressed_prob - prob, 2):+.2f}%)."
            ),
        }

    # Pricing (optional — requires current_price in input)
    current_price = item.get("current_price")
    if current_price is not None:
        effective_margin = (
            target_margin if target_margin is not None
            else item.get("target_margin", DEFAULT_TARGET_MARGIN)
        )
        pricing = calculate_optimal_price(
            net_sales, cogs, returns, discounts, current_price, effective_margin
        )
        if pricing:
            result["pricing"] = pricing

    return result


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

def aggregate_rows_by_product(rows, costs=None):
    """
    Aggregate date-level rows into one summary dict per product.

    costs (optional): list of {product, cogs_per_unit} or {product, total_cogs}.
    If provided, COGS is derived from the costs lookup instead of requiring it
    on every row. This lets clients send a raw sales export + a small cost table.
    """
    SUM_FIELDS = ["net_sales", "returns", "discounts", "units_sold", "cogs"]
    LAST_FIELDS = ["current_price", "cogs_increase_pct"]

    # Build costs lookup: product name (lowercased) → cost dict
    costs_lookup = {}
    if costs:
        for c in costs:
            name = str(c.get("product", "")).strip().lower()
            if name:
                costs_lookup[name] = c

    totals = {}
    for row in rows:
        name = str(row.get("product", "Unknown")).strip()
        key = name.lower()
        if key not in totals:
            totals[key] = {f: 0.0 for f in SUM_FIELDS}
            totals[key].update({f: None for f in LAST_FIELDS})
            totals[key]["product"] = name
        for f in SUM_FIELDS:
            if row.get(f) is not None:
                totals[key][f] += float(row[f])
        for f in LAST_FIELDS:
            if row.get(f) is not None:
                totals[key][f] = row[f]

    results = []
    skipped = []
    for key, item in totals.items():
        # Apply costs lookup if inline cogs not supplied via rows
        if item["cogs"] == 0.0 and key in costs_lookup:
            c = costs_lookup[key]
            if c.get("cogs_per_unit") is not None and item["units_sold"] > 0:
                item["cogs"] = float(c["cogs_per_unit"]) * item["units_sold"]
            elif c.get("total_cogs") is not None:
                item["cogs"] = float(c["total_cogs"])

        # Require cogs to proceed
        if item["cogs"] == 0.0:
            skipped.append(item["product"])
            continue

        # Derive profit from aggregated figures
        item["profit"] = item["net_sales"] - item["cogs"]

        # Clean up: remove optional zero/None fields except core ones
        core = {"product", "net_sales", "cogs", "profit", "returns", "discounts"}
        clean = {k: v for k, v in item.items()
                 if k in core or (v is not None and v != 0.0)}
        results.append(clean)

    return results, skipped


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "name": "Sales Profitability API",
        "endpoints": {
            "POST /predict": "Calculate profit probability for one or multiple products",
            "POST /rank": "Same as /predict but returns products sorted by overall_score",
            "POST /analyse-csv": "Accept raw date-level rows, aggregate by product, return ranked analysis",
            "GET /healthz": "Health check",
            "POST /admin/clients": "Create a new client key (requires X-Admin-Key)",
            "GET /admin/clients": "List all clients (requires X-Admin-Key)",
            "POST /admin/clients/<id>/revoke": "Revoke a client key (requires X-Admin-Key)",
            "POST /admin/clients/<id>/restore": "Restore a revoked client key (requires X-Admin-Key)",
        }
    })


@app.route("/predict", methods=["POST"])
@require_api_key
@limiter.limit("30 per minute")
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if isinstance(data, list):
        return jsonify({"data": [build_result(item) for item in data]})

    target_margin = data.get("target_margin")
    return jsonify({"data": build_result(data, target_margin)})


@app.route("/rank", methods=["POST"])
@require_api_key
@limiter.limit("30 per minute")
def rank():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if isinstance(data, dict):
        products = data.get("products", [])
        target_margin = data.get("target_margin")
    else:
        products = data
        target_margin = None

    results = [build_result(item, target_margin) for item in products]
    ranked = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return jsonify({"data": ranked})


@app.route("/analyse-csv", methods=["POST"])
@require_api_key
@limiter.limit("30 per minute")
def analyse_csv():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if isinstance(data, list):
        rows = data
        costs = None
        target_margin = None
    else:
        rows = data.get("rows", [])
        costs = data.get("costs")
        target_margin = data.get("target_margin")

    if not rows:
        return jsonify({"error": "No rows provided. Send an array of date-level rows."}), 400

    aggregated, skipped = aggregate_rows_by_product(rows, costs)

    if not aggregated:
        return jsonify({
            "error": "No products could be analysed. Provide COGS via inline rows or the 'costs' lookup.",
            "skipped": skipped,
        }), 400

    results = [build_result(item, target_margin) for item in aggregated]
    ranked = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1

    meta = {
        "rows_received": len(rows),
        "products_found": len(ranked),
    }
    if skipped:
        meta["skipped_products"] = skipped
        meta["skip_reason"] = "No COGS data found. Add these products to the 'costs' array."

    return jsonify({"data": ranked, "meta": meta})


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.route("/admin/dashboard", methods=["GET"])
def admin_dashboard():
    return render_template("admin.html")


@app.route("/admin/clients", methods=["POST"])
@require_admin_key
def create_client():
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return jsonify({"error": "A client name is required"}), 400

    new_key = secrets.token_hex(32)
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO clients (name, api_key, active, created_at) VALUES (?, ?, 1, ?)",
            (name, new_key, _now())
        )
        client_id = cursor.lastrowid
        conn.commit()

    return jsonify({
        "message": f"Client '{name}' created. Store this key — it won't be shown again.",
        "client": {"id": client_id, "name": name, "api_key": new_key, "active": True}
    }), 201


@app.route("/admin/clients", methods=["GET"])
@require_admin_key
def list_clients():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, active, created_at, last_used, request_count FROM clients ORDER BY id"
        ).fetchall()
    clients = [dict(row) for row in rows]
    for c in clients:
        c["active"] = bool(c["active"])
    return jsonify({"data": clients})


@app.route("/admin/clients/<int:client_id>/revoke", methods=["POST"])
@require_admin_key
def revoke_client(client_id):
    with get_db() as conn:
        row = conn.execute("SELECT name FROM clients WHERE id = ?", (client_id,)).fetchone()
        if not row:
            return jsonify({"error": "Client not found"}), 404
        conn.execute("UPDATE clients SET active = 0 WHERE id = ?", (client_id,))
        conn.commit()
    return jsonify({"message": f"Client '{row['name']}' revoked. Their key will be rejected immediately."})


@app.route("/admin/clients/<int:client_id>/restore", methods=["POST"])
@require_admin_key
def restore_client(client_id):
    with get_db() as conn:
        row = conn.execute("SELECT name FROM clients WHERE id = ?", (client_id,)).fetchone()
        if not row:
            return jsonify({"error": "Client not found"}), 404
        conn.execute("UPDATE clients SET active = 1 WHERE id = ?", (client_id,))
        conn.commit()
    return jsonify({"message": f"Client '{row['name']}' restored. Their key is active again."})


# ---------------------------------------------------------------------------
# Health & errors
# ---------------------------------------------------------------------------

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Too many requests — please slow down and try again shortly"}), 429


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port)
