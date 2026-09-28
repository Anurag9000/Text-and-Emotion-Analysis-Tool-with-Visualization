from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training_control import no_trainable_surface_v1 as authority


def make_required_tree(root: Path) -> None:
    for relative in authority.REQUIRED_INFERENCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# retained inference fixture\n", encoding="utf-8")


class InferenceOnlyAuthorityTests(unittest.TestCase):
    def test_live_repository_accounts_for_canonical_python_and_text_sources(self):
        payload = authority.audit()
        self.assertTrue(payload["complete"], payload["unresolved"])
        self.assertFalse(payload["missing_required_inference_files"])
        self.assertIn("Sentiment Analysis.py", payload["retained_python_files"])
        self.assertIn("Models.txt", payload["retained_text_files"])
        self.assertIn("inference_device_policy.py", payload["retained_python_files"])

    def test_empty_or_partial_repository_cannot_receive_inference_only_certificate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(authority, "ROOT", root):
                payload = authority.audit()
                self.assertFalse(payload["complete"])
                self.assertEqual(
                    payload["unresolved"][0]["type"], "required_inference_source_missing"
                )

    def test_code_like_training_primitive_in_txt_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_required_tree(root)
            (root / "Models.txt").write_text(
                "optimizer = torch.optim.Adam(model.parameters())\n",
                encoding="utf-8",
            )
            with mock.patch.object(authority, "ROOT", root):
                payload = authority.audit()
            self.assertFalse(payload["complete"])
            self.assertTrue(
                any(row["kind"] == "textual_torch_optim"
                    for row in payload["training_findings"])
            )

    def test_prose_use_of_training_word_does_not_create_false_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_required_tree(root)
            (root / "README.md").write_text(
                "This repository does not perform model training or benchmarking.\n",
                encoding="utf-8",
            )
            with mock.patch.object(authority, "ROOT", root):
                payload = authority.audit()
            self.assertTrue(payload["complete"], payload["unresolved"])

    def test_python_optimizer_call_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_required_tree(root)
            (root / "new_trainer.py").write_text(
                "class Model:\n"
                "    def fit(self, data):\n"
                "        return data\n"
                "model = Model()\n"
                "model.fit([1])\n",
                encoding="utf-8",
            )
            with mock.patch.object(authority, "ROOT", root):
                payload = authority.audit()
            self.assertFalse(payload["complete"])
            self.assertTrue(
                any(row["kind"] == "training_call" and row["symbol"] == "fit"
                    for row in payload["training_findings"])
            )


if __name__ == "__main__":
    unittest.main()
