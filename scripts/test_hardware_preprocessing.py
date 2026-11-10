"""Synthetic-only regression checks; never executes notebook experiment cells."""
from pathlib import Path
import ast
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hardware_preprocessing", ROOT / "qsvm_on_wdbc/hardware_preprocessing.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HardwarePreprocessingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(60, 8))
        self.y = np.tile([0, 1], 30)
        self.X[:, 0] += self.y * 2
        self.options = {"n_features_to_select": 3,
                        "estimator_params": {"solver": "lbfgs", "max_iter": 500,
                                             "random_state": 42}}

    def test_evaluation_outliers_cannot_change_fitted_selection_or_training(self):
        train_a, _, audit_a = module.prepare_partition(
            self.X, np.zeros((4, 8)), self.y, **self.options)
        train_b, _, audit_b = module.prepare_partition(
            self.X, np.full((4, 8), 1e12), self.y, **self.options)
        np.testing.assert_array_equal(train_a, train_b)
        self.assertEqual(audit_a["selected_indices"], audit_b["selected_indices"])
        self.assertFalse(audit_a["evaluation_rows_in_fit"])
        self.assertFalse(audit_a["evaluation_labels_in_fit"])

    def test_scaler_and_supervised_selector_receive_only_inner_training(self):
        scaler_fits, selector_fits, scopes = [], [], []
        original_scaler, original_selector = module.StandardScaler, module.RFE

        class SpyScaler(original_scaler):
            def fit(self, X, y=None, sample_weight=None):
                scaler_fits.append(np.array(X))
                return super().fit(X, y, sample_weight=sample_weight)

        class SpySelector(original_selector):
            def fit(self, X, y, **kwargs):
                selector_fits.append((np.array(X), np.array(y)))
                return super().fit(X, y, **kwargs)

        def kernels(train, evaluation, scope):
            scopes.append(scope)
            return rbf_kernel(train), rbf_kernel(evaluation, train)

        with patch.object(module, "StandardScaler", SpyScaler), patch.object(module, "RFE", SpySelector):
            C, score = module.nested_select_C(
                self.X, self.y, [0.1, 1.0], seed=42,
                preprocessing=self.options, kernel_builder=kernels, scope=2)
        splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(self.X, self.y))
        self.assertEqual(len(scaler_fits), 3)
        self.assertEqual(len(selector_fits), 3)
        self.assertEqual(scopes, [-2001, -2002, -2003])
        for i, (train, _) in enumerate(splits):
            np.testing.assert_array_equal(scaler_fits[i], self.X[train])
            np.testing.assert_array_equal(selector_fits[i][1], self.y[train])
        self.assertIn(C, [0.1, 1.0])
        self.assertTrue(np.isfinite(score))

    def test_changed_preprocessing_rejects_cached_kernel_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            module.prepare_partition(self.X, self.X[:3], self.y, audit_path=path)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                module.prepare_partition(self.X + 1, self.X[:3], self.y, audit_path=path)

    def test_candidate_tie_retains_first_C(self):
        def kernels(train, evaluation, scope):
            return np.ones((len(train), len(train))), np.ones((len(evaluation), len(train)))
        best, score = module.nested_select_C(
            self.X, self.y, [0.25, 1.0], seed=42, preprocessing={},
            kernel_builder=kernels)
        self.assertEqual(best, 0.25)
        self.assertEqual(score, 0.5)

    def test_all_notebook_nested_adapters_with_mock_kernels(self):
        notebook = json.loads((ROOT / "qsvm_on_wdbc/REAL_QPU.ipynb").read_text(encoding="utf-8"))
        for index in [13, 14, 16, 18, 20, 21, 23, 24, 25]:
            source = "".join(notebook["cells"][index]["source"])
            tree = ast.parse(source)
            self.assertNotIn(".fit(X_full)", source)
            self.assertNotIn(".fit(X_all)", source)
            self.assertNotIn("fit_transform(X_manual_all)", source)
            self.assertIn("prepare_partition(", source)
            self.assertIn(f'training_only_output_dir(OUTDIR, "cell{index}")', source)
            self.assertIsNone(notebook["cells"][index]["execution_count"])
            self.assertEqual(notebook["cells"][index]["outputs"], [])
            if index in [13, 14]:
                continue
            names = {"normalize_square_from_diag", "normalize_cross_from_diag",
                     "center_train_kernel", "center_test_kernel",
                     "_inner_training_kernels", "pick_best_C"}
            pure = ast.Module(body=[n for n in tree.body
                                   if isinstance(n, ast.FunctionDef) and n.name in names],
                              type_ignores=[])
            ns = {"np": np, "C_GRID": [0.1, 1.0], "RANDOM_SEED": 42,
                  "OUTDIR": tempfile.gettempdir(),
                  "nested_select_C": module.nested_select_C,
                  "kernel_square_upper": lambda A, details=None: rbf_kernel(A),
                  "kernel_diag": lambda A, details=None: np.ones(len(A)),
                  "kernel_cross": lambda A, B, details=None: rbf_kernel(A, B),
                  "kernel_train_square": lambda A, scope: rbf_kernel(A),
                  "kernel_test_diag": lambda A, scope: np.ones(len(A)),
                  "kernel_test_cross": lambda A, B, scope: rbf_kernel(A, B)}
            exec(compile(pure, f"mock_cell_{index}", "exec"), ns)
            # Pure functions only: no Qiskit imports, service authentication or jobs.
            with tempfile.TemporaryDirectory() as folder:
                ns["OUTDIR"] = folder
                C, score = ns["pick_best_C"](
                    self.X, self.y, preprocessing=self.options, scope=index)
                self.assertIn(C, [0.1, 1.0])
                self.assertTrue(np.isfinite(score))
            if index in [20, 21, 23, 24, 25]:
                K = rbf_kernel(self.X[:8]);cross = rbf_kernel(self.X[8:12], self.X[:8])
                result = ns["center_test_kernel"](cross, K)
                expected = cross - K.mean(axis=0) - cross.mean(axis=1, keepdims=True) + K.mean()
                np.testing.assert_allclose(result, expected)
                self.assertIn("center_test_kernel(K_te, K_tr_uncentered)", source)


if __name__ == "__main__":
    unittest.main()
