import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_from_directory)
from werkzeug.utils import secure_filename

import predictor

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
RESULTS_DIR = BASE / "results"
SITE_DATA = BASE / "site_data"
ADMIN_USER = os.environ.get("ADMIN_USER", "")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ---------- مساعدات ----------
def load_model_info():
    z = np.load(predictor.MODEL_PATH, allow_pickle=False)
    n_trees = len(z["offsets"]) - 1
    return {
        "file": predictor.MODEL_PATH.name,
        "size_kb": round(predictor.MODEL_PATH.stat().st_size / 1024, 1),
        "sha256": hashlib.sha256(predictor.MODEL_PATH.read_bytes()).hexdigest(),
        "n_trees": n_trees,
        "n_nodes": int(len(z["value"])),
        "features": [str(x) for x in z["features"]],
        "params": {
            "algorithm": "HistGradientBoostingClassifier (sklearn)",
            "max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 15,
            "l2_regularization": 2.0, "min_samples_leaf": 300,
            "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 8, "random_state": 42,
        },
        "target": "هل سيكون close بعد 30 دقيقة أعلى من close الآن؟",
        "threshold": predictor.THRESHOLD,
        "payout": predictor.PAYOUT,
        "break_even": round(predictor.BREAK_EVEN * 100, 2),
        "train_rows": 1577795,
        "train_end": "2026-08-31 22:58 UTC",
        "gap_minutes": 31,
    }


def load_data_info():
    with open(SITE_DATA / "data_info.json", encoding="utf-8") as f:
        return json.load(f)


def list_runs():
    runs = []
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            runs.append({"id": d.name, "files": sorted(p.name for p in d.glob("*.csv"))})
    return runs


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cumulative_svg(rows, width=760, height=260):
    """رسم SVG بسيط للربح التراكمي."""
    if not rows:
        return ""
    vals = [float(r["cumulative"]) for r in rows]
    lo, hi = min(0, min(vals)), max(0, max(vals))
    span = (hi - lo) or 1
    pad = 30
    n = len(vals)

    def xy(i, v):
        x = pad + (width - 2 * pad) * (i / max(n - 1, 1))
        y = pad + (height - 2 * pad) * (1 - (v - lo) / span)
        return round(x, 1), round(y, 1)

    pts = " ".join(f"{x},{y}" for x, y in (xy(i, v) for i, v in enumerate(vals)))
    zero_y = xy(0, 0)[1]
    labels = "".join(
        f'<text x="{xy(i, 0)[0]}" y="{height - 6}" font-size="10" text-anchor="middle" fill="#9ca3af">{rows[i]["month"]}</text>'
        for i in range(0, n, max(1, n // 8))
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">'
        f'<line x1="{pad}" x2="{width - pad}" y1="{zero_y}" y2="{zero_y}" stroke="#4b5563" stroke-dasharray="4"/>'
        f'<polyline points="{pts}" fill="none" stroke="#22c55e" stroke-width="2"/>'
        f'<text x="4" y="{pad}" font-size="10" fill="#9ca3af">{hi:.0f}</text>'
        f'<text x="4" y="{height - pad}" font-size="10" fill="#9ca3af">{lo:.0f}</text>'
        f"{labels}</svg>"
    )


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not ADMIN_USER or not ADMIN_PASS:
            abort(503, description="المنطقة الخاصة معطلة: ضع ADMIN_USER و ADMIN_PASS في البيئة.")
        auth = request.authorization
        if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
            return Response("يلزم تسجيل الدخول", 401,
                            {"WWW-Authenticate": 'Basic realm="BTC Model Admin"'})
        return fn(*args, **kwargs)
    return wrapper


# ---------- الصفحة 1: معلومات الموديل والبيانات ----------
@app.route("/")
def page_model():
    return render_template("model.html", model=load_model_info(), data=load_data_info(),
                           active="model")


# ---------- الصفحة 2: التحليل المباشر ----------
@app.route("/live")
def page_live():
    return render_template("live.html", active="live")


@app.route("/api/live")
def api_live():
    try:
        return jsonify(predictor.live_snapshot())
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 503


# ---------- الصفحة 3: النتائج ----------
@app.route("/results")
def page_results():
    runs = list_runs()
    run_id = request.args.get("run") or (runs[0]["id"] if runs else None)
    if run_id not in {r["id"] for r in runs}:
        abort(404)
    run_dir = RESULTS_DIR / run_id
    threshold = request.args.get("th", "0.58")
    if threshold not in ("0.58", "0.60"):
        threshold = "0.58"
    monthly_file = run_dir / f"walkforward_monthly_th{threshold}.csv"
    summary_file = run_dir / "halfyear_summary.csv"
    monthly = read_csv(monthly_file) if monthly_file.exists() else []
    summary = read_csv(summary_file) if summary_file.exists() else []

    totals = None
    if monthly:
        trades = sum(int(r["trades"]) for r in monthly)
        wins = sum(int(r["wins"]) for r in monthly)
        net = sum(float(r["net_units"]) for r in monthly)
        totals = {"trades": trades, "wins": wins,
                  "win_rate": round(wins / trades * 100, 2) if trades else 0,
                  "net": round(net, 1)}

    # مقارنة بين النسخ المتاحة (حالية وقديمة) لنفس العتبة
    comparison = []
    for r in runs:
        f = RESULTS_DIR / r["id"] / f"walkforward_monthly_th{threshold}.csv"
        if f.exists():
            rows = read_csv(f)
            t = sum(int(x["trades"]) for x in rows)
            w = sum(int(x["wins"]) for x in rows)
            comparison.append({"run": r["id"], "trades": t,
                               "win_rate": round(w / t * 100, 2) if t else 0,
                               "net": round(sum(float(x["net_units"]) for x in rows), 1)})

    return render_template(
        "results.html", active="results", runs=runs, run_id=run_id, threshold=threshold,
        monthly=monthly, summary=summary, totals=totals, comparison=comparison,
        chart=cumulative_svg(monthly), break_even=round(predictor.BREAK_EVEN * 100, 2),
    )


# ---------- المنطقة الخاصة: الموديل ----------
@app.route("/admin")
@require_admin
def page_admin():
    files = []
    for p in sorted(MODEL_DIR.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
            })
    return render_template("admin.html", active="admin", files=files,
                           message=request.args.get("msg"))


@app.route("/admin/download/<name>")
@require_admin
def admin_download(name):
    name = secure_filename(name)
    return send_from_directory(MODEL_DIR, name, as_attachment=True)


@app.route("/admin/upload", methods=["POST"])
@require_admin
def admin_upload():
    f = request.files.get("model")
    if not f or not f.filename.endswith(".npz"):
        return _msg("اختر ملف .npz صالحاً")
    name = secure_filename(f.filename)
    tmp = MODEL_DIR / f".upload_{name}"
    f.save(tmp)
    try:
        z = np.load(tmp, allow_pickle=False)
        needed = {"features", "offsets", "baseline", "feature_idx", "thr",
                  "left", "right", "value", "miss_left", "is_leaf"}
        if not needed.issubset(set(z.files)):
            raise ValueError("الملف لا يحتوي على كل حقول الموديل")
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return _msg(f"ملف غير صالح: {e}")
    backup = MODEL_DIR / f"{name}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    if (MODEL_DIR / name).exists():
        shutil.copy2(MODEL_DIR / name, backup)
    shutil.move(tmp, MODEL_DIR / name)
    predictor._model = None  # إعادة تحميل الموديل في الطلب القادم
    return _msg(f"تم رفع {name} بنجاح" + (f" (نسخة احتياطية: {backup.name})" if backup.exists() else ""))


def _msg(text):
    from urllib.parse import quote
    return Response(status=302, headers={"Location": f"/admin?msg={quote(text)}"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
