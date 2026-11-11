from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
REPO = PROJECT / "implementations"
SOURCE = ROOT / "source"
RECORDS = ROOT / "records"
TABLES = ROOT / "tables"
RUNS = ROOT / "runs"
IMPORTED = ROOT / "imported_existing"
TABLES.mkdir(exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(name: str, frame: pd.DataFrame) -> Path:
    path = TABLES / name
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clinical_metrics(predictions: pd.DataFrame) -> dict:
    y = (predictions["true_source_label"].to_numpy(int) == 0).astype(int)
    p = (predictions["predicted_source_label"].to_numpy(int) == 0).astype(int)
    scores = predictions["malignant_probability"].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "test_accuracy": float(accuracy_score(y, p)),
        "malignant_roc_auc": float(roc_auc_score(y, scores)),
        "sensitivity": float(recall_score(y, p)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "correct_count": int(tp + tn), "test_n": int(len(y)),
    }


def effective_key(row: pd.Series) -> tuple:
    width = int(row["n_features"])
    scaling = str(row["Scaling"])
    estimator = str(row["Estimator"])
    reps = int(row["Reps"])
    fmap = str(row["Feature_Map"])
    if fmap == "ZFeatureMap":
        return width, scaling, estimator, "Z", "", reps
    return width, scaling, estimator, "Pauli_ZZ", str(row["Entanglement"]), reps


def config_id(key: tuple) -> str:
    width, scaling, estimator, family, topology, reps = key
    scaler_token = "standard" if scaling == "Standardization" else "minmax"
    estimator_token = "lr" if estimator == "Logistic Regression" else "rf"
    encoder = "z" if family == "Z" else f"zz-{topology}"
    return f"fs-k{int(width):02d}-{scaler_token}-{estimator_token}-{encoder}-r{int(reps)}"


def load_archive():
    numeric = pd.read_csv(SOURCE / "historical" / "quantum_each_results.csv")
    text = pd.read_csv(SOURCE / "historical" / "quantum_each_results.csv", dtype=str, keep_default_na=False)
    mapping = pd.read_csv(SOURCE / "historical" / "ARCHIVED_552_CONFIGURATION_ROWS.csv")
    numeric.insert(0, "source_row_id", np.arange(1, len(numeric) + 1))
    text.insert(0, "source_row_id", np.arange(1, len(text) + 1))
    joined = numeric.merge(
        mapping[["archive_row_1_based", "identity_group", "canonical_gate_sha256"]].rename(columns={"archive_row_1_based": "source_row_id"}),
        on="source_row_id", validate="one_to_one",
    )
    groups = {}
    for key, group in joined.groupby(joined.apply(effective_key, axis=1), sort=False):
        group = group.sort_values("source_row_id", kind="stable")
        groups[key] = group
    return numeric, text, joined, groups


def result_from_artifact(metadata: dict, artifact: Path, origin: str, calibration_seed) -> dict:
    predictions = pd.read_csv(artifact / "test_predictions.csv")
    split = json.loads((SOURCE / "historical" / "split_idx.json").read_text(encoding="utf-8"))
    if predictions["sample_index"].tolist() != split["test_idx"]:
        raise RuntimeError(f"Test order mismatch: {artifact}")
    if len(predictions) != 114 or predictions["sample_index"].nunique() != 114:
        raise RuntimeError(f"Test count mismatch: {artifact}")
    if int((predictions["true_source_label"] == 0).sum()) != 42 or int((predictions["true_source_label"] == 1).sum()) != 72:
        raise RuntimeError(f"Class count mismatch: {artifact}")
    recomputed = clinical_metrics(predictions)
    saved = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    corrected = saved["corrected"]
    errors = []
    mapping = {"test_accuracy": "accuracy", "malignant_roc_auc": "roc_auc", "sensitivity": "sensitivity", "specificity": "specificity", "precision": "precision", "f1": "f1"}
    for local, stored in mapping.items():
        errors.append(abs(float(recomputed[local]) - float(corrected[stored])))
    if max(errors) > 1e-12:
        raise RuntimeError(f"Metric mismatch: {artifact}")
    features = json.loads((artifact / "features.json").read_text(encoding="utf-8"))
    selected_names = features.get("selected_feature_names", features.get("corrected_selected_feature_names", []))
    runtime = saved["runtime"]
    workers = 1
    if origin == "new_20261107":
        width = int(metadata["width"])
        settings_name = f"execution_settings_k{width}.json" if width in (11, 17) else "execution_settings.json"
        settings_path = RECORDS / settings_name
        if settings_path.exists():
            workers = int(json.loads(settings_path.read_text(encoding="utf-8"))["selected_workers"])
    row = {
        **metadata,
        "selected_feature_names": " | ".join(selected_names),
        "feature_selection_changed_from_archive": bool(features.get("selection_changed_from_archive", features.get("selection_changed", False))),
        "svc_settings": "C=1.0; probability=True; class_weight=None; shrinking=True; tol=0.001",
        "calibration_seed": calibration_seed,
        "evaluation_protocol": "fixed ordered seed-42 split; 455 train / 114 test; train-only scaler and RFE",
        "result_origin": origin,
        "train_accuracy": float(corrected.get("train_accuracy", np.nan)),
        **recomputed,
        "runtime_seconds": float(runtime["total_seconds"]),
        "started_at": runtime.get("started_at", ""),
        "completed_at": runtime.get("completed_at", ""),
        "workers": workers,
        "threads_per_worker": int(runtime.get("threads_per_worker", runtime.get("threads_per_run", 4))),
        "retry_count": int(runtime.get("retry_number", 0)),
        "benchmark_run": bool(runtime.get("benchmark_run", False)),
        "artifact_path": str(artifact.relative_to(ROOT)),
        "test_predictions_path": str((artifact / "test_predictions.csv").relative_to(ROOT)),
        "test_predictions_sha256": sha256_file(artifact / "test_predictions.csv"),
        "verification_status": "verified",
        "metric_recomputation_max_abs_error": max(errors),
    }
    return row


def build_fixed_results(archive: pd.DataFrame, groups: dict) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "execution_manifest.csv", keep_default_na=False)
    existing = pd.read_csv(RECORDS / "existing_results_inventory.csv", keep_default_na=False)
    existing_map = {row.configuration_id: row for row in existing.itertuples(index=False)}
    rows = []
    for item in manifest.to_dict(orient="records"):
        cid = item["configuration_id"]
        family = "ZFeatureMap" if item["execution_constructor"] == "ZFeatureMap" else "Pauli/ZZ equivalent"
        metadata = {
            "configuration_id": cid,
            "width": int(item["width"]),
            "scaler": item["scaler"],
            "rfe_estimator": item["rfe_estimator"],
            "encoder_family": family,
            "original_feature_map_label": item["nominal_feature_map_labels"],
            "execution_constructor": item["execution_constructor"],
            "topology": item["topology"],
            "repetitions": int(item["repetitions"]),
            "source_row_ids": item["source_row_ids"],
            "representative_archived_source_row_id": int(item["representative_source_row_id"]) if str(item["representative_source_row_id"]).strip() else np.nan,
        }
        if item["execution_action"] == "reuse_verified":
            prior = existing_map[cid]
            artifact = ROOT / prior.imported_artifact_path
            metadata["original_feature_map_label"] = str(archive.iloc[int(prior.source_row_id) - 1]["Feature_Map"])
            metadata["representative_archived_source_row_id"] = int(prior.source_row_id)
            row = result_from_artifact(metadata, artifact, "reused_prior_verified", "uncontrolled_None")
        else:
            artifact = RUNS / f"k{int(item['width']):02d}" / cid
            row = result_from_artifact(metadata, artifact, "new_20261107", 20261107)
        rows.append(row)
    manifest_widths = set(manifest["width"].astype(int).unique())
    broader = existing[~existing["width"].astype(int).isin(manifest_widths)]
    for prior in broader.itertuples(index=False):
        source_row = archive.iloc[int(prior.source_row_id) - 1]
        key = effective_key(source_row)
        group = groups[key]
        metadata = {
            "configuration_id": config_id(key),
            "width": int(source_row["n_features"]),
            "scaler": "StandardScaler" if source_row["Scaling"] == "Standardization" else "MinMaxScaler",
            "rfe_estimator": source_row["Estimator"],
            "encoder_family": "ZFeatureMap" if source_row["Feature_Map"] == "ZFeatureMap" else "Pauli/ZZ equivalent",
            "original_feature_map_label": source_row["Feature_Map"],
            "execution_constructor": source_row["Feature_Map"],
            "topology": "" if source_row["Feature_Map"] == "ZFeatureMap" else source_row["Entanglement"],
            "repetitions": int(source_row["Reps"]),
            "source_row_ids": "|".join(map(str, group["source_row_id"].astype(int).tolist())),
            "representative_archived_source_row_id": int(prior.source_row_id),
        }
        rows.append(result_from_artifact(metadata, ROOT / prior.imported_artifact_path, "reused_prior_verified", "uncontrolled_None"))
    fixed = pd.DataFrame(rows)
    expected = int(manifest["configuration_id"].nunique() + broader["configuration_id"].nunique())
    if len(fixed) != expected or fixed["configuration_id"].nunique() != expected:
        raise RuntimeError(f"Expected {expected} unique fixed results, found {len(fixed)}/{fixed['configuration_id'].nunique()}")
    fixed = fixed.sort_values(["width", "scaler", "rfe_estimator", "encoder_family", "topology", "repetitions", "configuration_id"], kind="stable").reset_index(drop=True)
    return fixed


def build_before_after(fixed: pd.DataFrame, archive: pd.DataFrame, archive_text: pd.DataFrame, groups: dict) -> pd.DataFrame:
    rows = []
    for result in fixed.itertuples(index=False):
        if pd.isna(result.representative_archived_source_row_id):
            rows.append({
                "comparison_scope": "scaled_corrected_no_archived_mapping",
                "configuration_id": result.configuration_id,
                "width": result.width,
                "scaler": result.scaler,
                "rfe_estimator": result.rfe_estimator,
                "encoder_family": result.encoder_family,
                "original_feature_map_label": result.original_feature_map_label,
                "topology": result.topology,
                "repetitions": result.repetitions,
                "archived_source_row_id_used": "",
                "all_archived_alias_source_row_ids": "",
                "archived_accuracy_reported": np.nan,
                "archived_correct_count": np.nan,
                "corrected_accuracy": result.test_accuracy,
                "corrected_correct_count": result.correct_count,
                "accuracy_change_percentage_points": np.nan,
                "archived_roc_auc_reported_text": "",
                "archived_roc_auc_used": np.nan,
                "archived_alias_roc_auc_min": np.nan,
                "archived_alias_roc_auc_max": np.nan,
                "archived_alias_roc_auc_conflict": np.nan,
                "corrected_malignant_roc_auc": result.malignant_roc_auc,
                "roc_auc_change_vs_used_source": np.nan,
                "auc_comparability": "no archived source-row mapping; corrected score available but difference unavailable",
                "calibration_seed": result.calibration_seed,
                "artifact_path": result.artifact_path,
                "provenance": "effective k=17 grid cell absent from the 552-row archived mapping",
            })
            continue
        source_id = int(result.representative_archived_source_row_id)
        archived = archive.iloc[source_id - 1]
        archived_text_row = archive_text.iloc[source_id - 1]
        group = groups[effective_key(archived)]
        auc_values = group["QSVC_AUC"].astype(float)
        archived_correct = int(archived["QSVC_TP"]) + int(archived["QSVC_TN"])
        rows.append({
            "comparison_scope": "scaled_corrected_match",
            "configuration_id": result.configuration_id,
            "width": result.width,
            "scaler": result.scaler,
            "rfe_estimator": result.rfe_estimator,
            "encoder_family": result.encoder_family,
            "original_feature_map_label": result.original_feature_map_label,
            "topology": result.topology,
            "repetitions": result.repetitions,
            "archived_source_row_id_used": source_id,
            "all_archived_alias_source_row_ids": result.source_row_ids,
            "archived_accuracy_reported": float(archived["QSVC_Test_Acc"]),
            "archived_correct_count": archived_correct,
            "corrected_accuracy": result.test_accuracy,
            "corrected_correct_count": result.correct_count,
            "accuracy_change_percentage_points": 100.0 * (int(result.correct_count) - archived_correct) / 114.0,
            "archived_roc_auc_reported_text": archived_text_row["QSVC_AUC"],
            "archived_roc_auc_used": float(archived["QSVC_AUC"]),
            "archived_alias_roc_auc_min": float(auc_values.min()),
            "archived_alias_roc_auc_max": float(auc_values.max()),
            "archived_alias_roc_auc_conflict": bool(auc_values.nunique() != 1),
            "corrected_malignant_roc_auc": result.malignant_roc_auc,
            "roc_auc_change_vs_used_source": result.malignant_roc_auc - float(archived["QSVC_AUC"]),
            "auc_comparability": "numeric orientation comparable; calibration randomness differs for new runs" if result.result_origin == "new_20261107" else "numeric orientation comparable; reused calibration seed was uncontrolled",
            "calibration_seed": result.calibration_seed,
            "artifact_path": result.artifact_path,
            "provenance": "552-row scaled archive matched by complete effective configuration",
        })
    no_scaling = pd.read_csv(SOURCE / "historical" / "quantum_no_scaling_qsvc_results.csv", dtype={"QSVC_AUC": str})
    for index, item in no_scaling.iterrows():
        correct = int(item["QSVC_TP"]) + int(item["QSVC_TN"])
        rows.append({
            "comparison_scope": "archived_no_scaling_reference_only",
            "configuration_id": f"no-scale-archive-{index + 1:03d}",
            "width": int(item["n_features"]),
            "scaler": "None",
            "rfe_estimator": "Not recorded / not applicable",
            "encoder_family": item["Feature_Map"],
            "original_feature_map_label": item["Feature_Map"],
            "topology": item["Entanglement"],
            "repetitions": int(item["Reps"]),
            "archived_source_row_id_used": f"no-scaling-{index + 1:03d}",
            "all_archived_alias_source_row_ids": "",
            "archived_accuracy_reported": float(item["QSVC_Test_Acc"]),
            "archived_correct_count": correct,
            "corrected_accuracy": np.nan,
            "corrected_correct_count": np.nan,
            "accuracy_change_percentage_points": np.nan,
            "archived_roc_auc_reported_text": str(item["QSVC_AUC"]),
            "archived_roc_auc_used": float(item["QSVC_AUC"]),
            "archived_alias_roc_auc_min": np.nan,
            "archived_alias_roc_auc_max": np.nan,
            "archived_alias_roc_auc_conflict": False,
            "corrected_malignant_roc_auc": np.nan,
            "roc_auc_change_vs_used_source": np.nan,
            "auc_comparability": "no training-only corrected no-scaling counterpart; reference only",
            "calibration_seed": "uncontrolled / not recorded",
            "artifact_path": "source/historical/quantum_no_scaling_qsvc_results.csv",
            "provenance": "legacy no-scaling QSVC aggregate; not merged with scaled corrected grid",
        })
    return pd.DataFrame(rows)


def all_expected_keys(width: int):
    for scaling in ("Standardization", "Normalization"):
        for estimator in ("Logistic Regression", "Random Forest"):
            for reps in range(1, 5):
                yield width, scaling, estimator, "Z", "", reps
            for topology in ("linear", "full", "circular"):
                for reps in range(1, 5):
                    yield width, scaling, estimator, "Pauli_ZZ", topology, reps


def build_grid_coverage(fixed: pd.DataFrame, groups: dict) -> pd.DataFrame:
    corrected = {row.configuration_id: row for row in fixed.itertuples(index=False)}
    rows = []
    for width in (4, 7, 11, 17):
        for order, key in enumerate(all_expected_keys(width), start=1):
            cid = config_id(key)
            group = groups.get(key)
            result = corrected.get(cid)
            if result is not None:
                status = "completed_new" if result.result_origin == "new_20261107" else "completed_reused"
                verification = "verified"
                artifact = result.artifact_path
            elif group is None:
                status = "archive_mapping_missing"
                verification = "not available"
                artifact = ""
            else:
                status = "not_run_out_of_scope"
                verification = "archived mapping only"
                artifact = ""
            scaling = key[1]
            estimator = key[2]
            family = key[3]
            rows.append({
                "width": width,
                "grid_order_within_width": order,
                "configuration_id": cid,
                "scaler": "StandardScaler" if scaling == "Standardization" else "MinMaxScaler",
                "rfe_estimator": estimator,
                "effective_encoder_family": "ZFeatureMap" if family == "Z" else "Pauli/ZZ equivalent",
                "topology": key[4],
                "repetitions": key[5],
                "archived_mapping_available": group is not None,
                "archived_source_row_ids": "" if group is None else "|".join(map(str, group["source_row_id"].astype(int).tolist())),
                "corrected_result_available": result is not None,
                "status": status,
                "verification_status": verification,
                "artifact_path": artifact,
            })
    frame = pd.DataFrame(rows)
    if len(frame) != 256:
        raise RuntimeError("Grid coverage must have 256 expected rows across four widths")
    return frame


def build_best(fixed: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for width, group in fixed.groupby("width", sort=True):
        max_correct = int(group["correct_count"].max())
        tied = group[group["correct_count"] == max_correct].sort_values(["malignant_roc_auc", "configuration_id"], ascending=[False, True], kind="stable")
        complete = bool((coverage[(coverage["width"] == width)]["corrected_result_available"]).all())
        for display_order, item in enumerate(tied.itertuples(index=False), start=1):
            rows.append({
                "width": int(width),
                "accuracy_tie_count": len(tied),
                "tie_display_order": display_order,
                "configuration_id": item.configuration_id,
                "scaler": item.scaler,
                "rfe_estimator": item.rfe_estimator,
                "encoder_family": item.encoder_family,
                "topology": item.topology,
                "repetitions": item.repetitions,
                "corrected_accuracy": item.test_accuracy,
                "correct_count_out_of_114": item.correct_count,
                "malignant_roc_auc": item.malignant_roc_auc,
                "calibration_seed": item.calibration_seed,
                "grid_coverage_complete": complete,
                "completed_corrected_configurations": int(len(group)),
                "expected_effective_configurations": 64,
                "ranking_rule": "primary: exact correct count / 114; report all accuracy ties; AUC only orders display and is not a tie-break because calibration randomness differs",
                "artifact_path": item.artifact_path,
            })
    return pd.DataFrame(rows)


def build_aliases(joined: pd.DataFrame, groups: dict) -> pd.DataFrame:
    rows = []
    for item in joined.itertuples(index=False):
        series = pd.Series(item._asdict())
        key = effective_key(series)
        group = groups[key]
        rows.append({
            "source_row_id": int(item.source_row_id),
            "width": int(item.n_features),
            "scaling_label": item.Scaling,
            "rfe_estimator": item.Estimator,
            "nominal_feature_map_label": item.Feature_Map,
            "nominal_topology_label": item.Entanglement,
            "repetitions": int(item.Reps),
            "effective_configuration_id": config_id(key),
            "effective_encoder_family": "ZFeatureMap" if key[3] == "Z" else "Pauli/ZZ equivalent",
            "effective_topology": key[4],
            "identity_group": item.identity_group,
            "canonical_gate_sha256": item.canonical_gate_sha256,
            "alias_group_source_row_ids": "|".join(map(str, group["source_row_id"].astype(int).tolist())),
            "accuracy_consistent_with_aliases": bool(group["QSVC_Test_Acc"].astype(float).nunique() == 1),
            "f1_consistent_with_aliases": bool(group["QSVC_F1"].astype(float).nunique() == 1),
            "archived_auc_alias_conflict": bool(group["QSVC_AUC"].astype(float).nunique() != 1),
            "alias_rule": "Z ignores nominal topology; PauliFeatureMap[Z,ZZ] and ZZFeatureMap are equivalent at the same topology/reps/settings",
        })
    return pd.DataFrame(rows)


def build_run_log(fixed: pd.DataFrame) -> pd.DataFrame:
    settings = json.loads((RECORDS / "execution_settings.json").read_text(encoding="utf-8"))
    rows = []
    for item in fixed.itertuples(index=False):
        rows.append({
            "configuration_id": item.configuration_id,
            "width": item.width,
            "result_origin": item.result_origin,
            "started_at": item.started_at,
            "completed_at": item.completed_at,
            "runtime_seconds": item.runtime_seconds,
            "worker_processes_for_batch": item.workers,
            "threads_per_run": item.threads_per_worker,
            "benchmark_run": item.benchmark_run,
            "failure_count": 0,
            "retry_count": item.retry_count,
            "checkpoint_path": item.artifact_path + "/metrics.json",
            "checkpoint_verified": True,
            "calibration_seed": item.calibration_seed,
        })
    return pd.DataFrame(rows)


def build_repeated_cv() -> pd.DataFrame:
    path = SOURCE / "repeated_cv" / "repeated_oof_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for fold in payload["folds"]:
        metrics = fold["metrics"]
        timing = fold["timing"]
        rows.append({
            "record_type": "outer_fold",
            "procedure_scope": payload["scope"],
            "procedure_status": payload["status"],
            "repeat_id": int(fold["repeat_id"]),
            "outer_fold_index": int(fold["outer_fold_index"]),
            "selected_configuration_id": fold["selected_configuration_id"],
            "run_id": metrics.get("run_id", ""),
            "split_manifest_sha256": metrics.get("split_manifest_sha256", ""),
            "pipeline_config_sha256": metrics.get("pipeline_config_sha256", ""),
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
            "TP": metrics["tp"], "FN": metrics["fn"], "FP": metrics["fp"], "TN": metrics["tn"],
            "wall_seconds": timing["wall_seconds"],
            "warning_count": fold["warning_count"],
            "source_path": str(path.relative_to(ROOT)),
        })
    for repeat in payload["repeat_level"]:
        metrics = repeat["metrics"]
        rows.append({
            "record_type": "repeat_summary",
            "procedure_scope": payload["scope"],
            "procedure_status": payload["status"],
            "repeat_id": int(repeat["repeat_id"]),
            "outer_fold_index": "all_5",
            "selected_configuration_id": "nested selection varies by fold",
            "run_id": "scientific-qsvc-nested-20261015",
            "split_manifest_sha256": "",
            "pipeline_config_sha256": "",
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
            "TP": metrics["tp"], "FN": metrics["fn"], "FP": metrics["fp"], "TN": metrics["tn"],
            "wall_seconds": np.nan,
            "warning_count": np.nan,
            "source_path": str(path.relative_to(ROOT)),
        })
    mean = payload["arithmetic_mean_of_repeat_metrics"]
    rows.append({
        "record_type": "arithmetic_mean_of_repeat_metrics",
        "procedure_scope": payload["scope"],
        "procedure_status": payload["status"],
        "repeat_id": "1-3",
        "outer_fold_index": "all_15",
        "selected_configuration_id": "nested selection varies by fold",
        "run_id": "scientific-qsvc-nested-20261015",
        "split_manifest_sha256": "",
        "pipeline_config_sha256": "",
        "accuracy": mean["accuracy"], "roc_auc": mean["roc_auc"], "average_precision": mean["average_precision"],
        "sensitivity": mean["sensitivity"], "specificity": mean["specificity"], "precision": mean["precision"],
        "f1": mean["f1"], "mcc": mean["mcc"],
        "TP": np.nan, "FN": np.nan, "FP": np.nan, "TN": np.nan,
        "wall_seconds": np.nan, "warning_count": np.nan,
        "source_path": str(path.relative_to(ROOT)),
    })
    return pd.DataFrame(rows)


def build_provenance() -> pd.DataFrame:
    repo = json.loads((RECORDS / "repository_start_state.json").read_text(encoding="utf-8"))
    split = json.loads((RECORDS / "split_verification.json").read_text(encoding="utf-8"))
    machine = json.loads((RECORDS / "machine_preflight.json").read_text(encoding="utf-8"))
    settings = json.loads((RECORDS / "execution_settings.json").read_text(encoding="utf-8"))
    kernel = json.loads((RECORDS / "kernel_equivalence_audit.json").read_text(encoding="utf-8"))
    settings_k11 = json.loads((RECORDS / "execution_settings_k11.json").read_text(encoding="utf-8"))
    completion_k11 = json.loads((RECORDS / "execution_completion_k11.json").read_text(encoding="utf-8"))
    kernel_k11 = json.loads((RECORDS / "kernel_equivalence_audit_k11.json").read_text(encoding="utf-8"))
    manifest_lock_k11 = json.loads((RECORDS / "k11_manifest_lock.json").read_text(encoding="utf-8"))
    settings_k17 = json.loads((RECORDS / "execution_settings_k17.json").read_text(encoding="utf-8"))
    completion_k17 = json.loads((RECORDS / "execution_completion_k17.json").read_text(encoding="utf-8"))
    manifest_lock_k17 = json.loads((RECORDS / "k17_manifest_lock.json").read_text(encoding="utf-8"))
    fast_k17 = json.loads((RECORDS / "k17_fast_simulator_equivalence.json").read_text(encoding="utf-8"))
    implementation_benchmark_k17 = json.loads((RECORDS / "k17_implementation_benchmark.json").read_text(encoding="utf-8"))
    workbook_backup = ROOT / "WDBC_QSVC_Training_Only_Master_pre_k11_20261107.xlsx"
    values = [
        ("output_directory", str(ROOT), "current execution"),
        ("repository_url", repo["remote_url"], "git remote origin"),
        ("repository_commit", repo["commit"], "git rev-parse HEAD"),
        ("repository_start_status_sha256", repo["status_sha256"], "git status --porcelain=v1 --untracked-files=all"),
        ("manifest_path", "execution_manifest.csv", "locked before model fitting"),
        ("manifest_sha256", sha256_file(ROOT / "execution_manifest.csv"), "sha256"),
        ("dataset_path", "source/data/wdbc_arrays.npz", "isolated copy"),
        ("dataset_sha256", sha256_file(SOURCE / "data" / "wdbc_arrays.npz"), "sha256"),
        ("split_path", "source/historical/split_idx.json", "ordered original split"),
        ("train_index_sha256", split["train_index_sha256"], "455 ordered indices"),
        ("test_index_sha256", split["test_index_sha256"], "114 ordered indices"),
        ("test_class_counts", "42 malignant label 0; 72 benign label 1", "verified"),
        ("positive_class", "malignant; source label 0", "clinical metrics"),
        ("fit_access", "scaler and RFE fit on 455 training rows; QSVC fit on transformed training rows and labels only", "verified per artifact"),
        ("new_probability_calibration_seed", 20261107, "locked before execution"),
        ("reused_probability_calibration_seed", "None / uncontrolled", "preserved prior values"),
        ("kernel_semantics", "analytic squared statevector fidelity; shots=None; blocked float64 Gram matrices", "unchanged scientific computation"),
        ("kernel_equivalence_tolerance", kernel["tolerance"], "FidelityStatevectorKernel and alias audit"),
        ("kernel_equivalence_max_error", max(item["max_abs_error_vs_FidelityStatevectorKernel"] for item in kernel["constructor_checks"]), "passed"),
        ("workers", settings["selected_workers"], settings["reason"]),
        ("threads_per_worker", settings["selected_threads_per_worker"], "no physical-core oversubscription"),
        ("package_versions", json.dumps(machine["packages"], sort_keys=True), "validated environment"),
        ("python_executable", machine["python_executable"], "validated environment"),
        ("scaled_archive", "source/historical/quantum_each_results.csv", "552 configuration rows"),
        ("no_scaling_archive", "source/historical/quantum_no_scaling_qsvc_results.csv", "48 aggregate reference rows; no corrected counterpart"),
        ("repeated_cv_source", "source/repeated_cv/repeated_oof_summary.json", "kept separate from fixed split"),
        ("selection_history_caveat", "fixed-split results are exploratory; the original test split informed historical configuration selection", "interpretation constraint"),
        ("scientific_scope", "training-only preprocessing sensitivity check and completion of the k=4/k=7/k=11/k=17 scaled effective grids", "not independent validation"),
        ("k11_manifest_path", "execution_manifest_k11.csv", "locked before the 60 pending k=11 evaluations"),
        ("k11_manifest_sha256", manifest_lock_k11["manifest_sha256"], "64 rows: 60 execute and 4 verified reuse"),
        ("k11_combined_manifest_sha256", manifest_lock_k11["combined_manifest_sha256"], "192 k=4/k=7/k=11 rows"),
        ("k11_kernel_equivalence_tolerance", kernel_k11["tolerance"], "k=11 constructor and Pauli/ZZ alias audit"),
        ("k11_kernel_equivalence_max_error", max(item["max_abs_error"] for item in kernel_k11["constructor_checks"]), "passed"),
        ("k11_benchmark_configuration", settings_k11["benchmark_configuration_id"], "authorized grid evaluation; retained as a result"),
        ("k11_benchmark_runtime_seconds", settings_k11["benchmark_result"]["runtime_seconds"], "measured before choosing concurrency"),
        ("k11_benchmark_peak_rss_bytes", settings_k11["benchmark_result"]["peak_observed_rss_bytes"], "resource observation"),
        ("k11_workers", settings_k11["selected_workers"], settings_k11["reason"]),
        ("k11_threads_per_worker", settings_k11["selected_threads_per_worker"], "four total BLAS threads on four physical cores"),
        ("k11_completion", f'{completion_k11["new_completed"]} new; {completion_k11["reused_completed"]} reused; {completion_k11["failed"]} failed', "64/64 effective configurations verified"),
        ("pre_k11_workbook_backup", workbook_backup.name, "backup made before editing the existing nine-sheet workbook"),
        ("pre_k11_workbook_backup_sha256", sha256_file(workbook_backup), "sha256"),
        ("k17_manifest_path", "execution_manifest_k17.csv", "locked before the 59 pending k=17 evaluations"),
        ("k17_manifest_sha256", manifest_lock_k17["manifest_sha256"], "64 rows: 59 execute and 5 verified reuse"),
        ("k17_archived_mapping_missing", manifest_lock_k17["archived_mapping_missing"], "corrected results retained; archived differences unavailable"),
        ("k17_fast_simulator_cases", fast_k17["cases"], "widths 4/7/11/17; repetitions and all supported entanglements"),
        ("k17_fast_simulator_max_kernel_error", fast_k17["max_absolute_kernel_error"], f'tolerance {fast_k17["tolerance"]}'),
        ("k17_fast_simulator_aggregate_speedup", fast_k17["aggregate_speedup"], "Qiskit reference time divided by exact vectorized simulator time"),
        ("k17_implementation_memory_reduction_fraction", implementation_benchmark_k17["estimated_array_memory_reduction_fraction"], "single-allocation train bank and streamed test kernel"),
        ("k17_selected_backend_gpu_supported", implementation_benchmark_k17["selected_backend_gpu_supported"], implementation_benchmark_k17["gpu_note"]),
        ("k17_benchmark_configuration", settings_k17["benchmark_configuration_id"], "authorized grid evaluation; retained as a result"),
        ("k17_benchmark_runtime_seconds", settings_k17["benchmark_result"]["runtime_seconds"], "measured before choosing concurrency"),
        ("k17_benchmark_peak_rss_bytes", settings_k17["benchmark_result"]["peak_observed_rss_bytes"], "resource observation"),
        ("k17_workers", settings_k17["selected_workers"], settings_k17["reason"]),
        ("k17_threads_per_worker", settings_k17["selected_threads_per_worker"], "bounded by four physical cores"),
        ("k17_completion", f'{completion_k17["new_completed"]} new; {completion_k17["reused_completed"]} reused; {completion_k17["failed"]} failed', "64/64 effective configurations verified"),
        ("k17_resume_command", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\resume_k17.ps1", "idempotent checkpoint-aware resume command"),
    ]
    source_inventory = pd.read_csv(RECORDS / "source_inventory.csv")
    for item in source_inventory.itertuples(index=False):
        values.append((f"source_sha256::{item.isolated_path}", item.sha256, item.source_path))
    return pd.DataFrame(values, columns=["provenance_item", "value", "verification_or_note"])


def run_git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    archive, archive_text, joined, groups = load_archive()
    fixed = build_fixed_results(archive, groups)
    before_after = build_before_after(fixed, archive, archive_text, groups)
    coverage = build_grid_coverage(fixed, groups)
    best = build_best(fixed, coverage)
    aliases = build_aliases(joined, groups)
    run_log = build_run_log(fixed)
    repeated = build_repeated_cv()
    provenance = build_provenance()
    summary_rows = []
    for width in (4, 7, 11, 17):
        results = fixed[fixed["width"] == width]
        grid = coverage[coverage["width"] == width]
        summary_rows.append({
            "width": width,
            "expected_effective_grid": 64,
            "archived_mapping_available": int(grid["archived_mapping_available"].sum()),
            "corrected_completed": len(results),
            "new_completed_this_execution": int((results["result_origin"] == "new_20261107").sum()),
            "reused_verified": int((results["result_origin"] == "reused_prior_verified").sum()),
            "pending_in_authorized_scope": int(64 - len(results)),
            "failed": 0,
            "coverage_complete": bool(grid["corrected_result_available"].all()),
            "verification_status": "verified_complete" if bool(grid["corrected_result_available"].all()) else "verified_partial_existing_evidence",
        })
    summary = pd.DataFrame(summary_rows)
    no_scaling = pd.read_csv(SOURCE / "historical" / "quantum_no_scaling_qsvc_results.csv")
    no_scaling_summary = no_scaling.loc[no_scaling.groupby("n_features")["QSVC_Test_Acc"].idxmax()].copy()
    no_scaling_summary.insert(0, "context", "archived no-scaling reference only; no training-only corrected counterpart")
    files = {
        "summary.csv": summary,
        "fixed_split_results.csv": fixed,
        "before_after.csv": before_after,
        "best_per_width.csv": best,
        "grid_coverage.csv": coverage,
        "encoder_aliases.csv": aliases,
        "run_log.csv": run_log,
        "repeated_cv.csv": repeated,
        "provenance.csv": provenance,
        "no_scaling_summary.csv": no_scaling_summary,
    }
    for name, frame in files.items():
        write_csv(name, frame)
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    start = json.loads((RECORDS / "repository_start_state.json").read_text(encoding="utf-8"))
    end = {
        "recorded_at": now_iso(),
        "repository": str(REPO),
        "remote_url": run_git("remote", "get-url", "origin"),
        "commit": run_git("rev-parse", "HEAD"),
        "status_porcelain": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }
    end["matches_start_commit"] = end["commit"] == start["commit"]
    end["matches_start_status"] = end["status_sha256"] == start["status_sha256"]
    write_json(RECORDS / "repository_end_state.json", end)
    validation = {
        "verified_at": now_iso(),
        "fixed_split_rows": len(fixed),
        "fixed_split_unique_configurations": int(fixed["configuration_id"].nunique()),
        "new_fixed_split_rows": int((fixed["result_origin"] == "new_20261107").sum()),
        "reused_fixed_split_rows": int((fixed["result_origin"] == "reused_prior_verified").sum()),
        "k4_corrected_rows": int((fixed["width"] == 4).sum()),
        "k7_corrected_rows": int((fixed["width"] == 7).sum()),
        "k11_corrected_rows": int((fixed["width"] == 11).sum()),
        "k17_corrected_rows": int((fixed["width"] == 17).sum()),
        "all_prediction_sets_114_rows": bool((fixed["test_n"] == 114).all()),
        "max_metric_recomputation_error": float(fixed["metric_recomputation_max_abs_error"].max()),
        "grid_coverage_rows": len(coverage),
        "k4_grid_completed": int(coverage[(coverage["width"] == 4) & coverage["corrected_result_available"]].shape[0]),
        "k7_grid_completed": int(coverage[(coverage["width"] == 7) & coverage["corrected_result_available"]].shape[0]),
        "k11_grid_completed": int(coverage[(coverage["width"] == 11) & coverage["corrected_result_available"]].shape[0]),
        "k17_grid_completed": int(coverage[(coverage["width"] == 17) & coverage["corrected_result_available"]].shape[0]),
        "before_after_corrected_matches": int((before_after["comparison_scope"] == "scaled_corrected_match").sum()),
        "before_after_corrected_without_archived_mapping": int((before_after["comparison_scope"] == "scaled_corrected_no_archived_mapping").sum()),
        "before_after_no_scaling_context_rows": int((before_after["comparison_scope"] == "archived_no_scaling_reference_only").sum()),
        "encoder_alias_rows": len(aliases),
        "repeated_cv_rows": len(repeated),
        "failures": 0,
        "retries": int(run_log["retry_count"].sum()),
        "repository_commit_matches_start": end["matches_start_commit"],
        "repository_status_matches_start": end["matches_start_status"],
        "passed": bool(
            len(fixed) == 256 and fixed["configuration_id"].nunique() == 256
            and (fixed["test_n"] == 114).all()
            and fixed["metric_recomputation_max_abs_error"].max() <= 1e-12
            and coverage[(coverage["width"] == 4) & coverage["corrected_result_available"]].shape[0] == 64
            and coverage[(coverage["width"] == 7) & coverage["corrected_result_available"]].shape[0] == 64
            and coverage[(coverage["width"] == 11) & coverage["corrected_result_available"]].shape[0] == 64
            and coverage[(coverage["width"] == 17) & coverage["corrected_result_available"]].shape[0] == 64
            and (before_after["comparison_scope"] == "scaled_corrected_no_archived_mapping").sum() == 5
            and end["matches_start_commit"] and end["matches_start_status"]
        ),
    }
    write_json(RECORDS / "consolidation_verification.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"Consolidation verification failed: {validation}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
