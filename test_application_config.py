from __future__ import annotations

import unittest
from pathlib import Path

from application_config import database_config_from_env


ROOT = Path(__file__).resolve().parent
APP_FILES = ("Sentiment Analysis.py", "Sentiment Analysis (Stable last code).py")


class ApplicationConfigTests(unittest.TestCase):
    def test_database_password_is_required_from_environment(self):
        with self.assertRaisesRegex(RuntimeError, "SENTIMENT_DB_PASSWORD"):
            database_config_from_env({})

    def test_non_secret_local_defaults_and_environment_overrides(self):
        config = database_config_from_env({"SENTIMENT_DB_PASSWORD": "fixture-secret"})
        self.assertEqual(config["host"], "localhost")
        self.assertEqual(config["user"], "root")
        self.assertEqual(config["database"], "SentimentAnalysis")
        self.assertEqual(config["password"], "fixture-secret")

        config = database_config_from_env({
            "SENTIMENT_DB_HOST": "db.internal",
            "SENTIMENT_DB_USER": "sentiment",
            "SENTIMENT_DB_PASSWORD": "another-fixture",
            "SENTIMENT_DB_NAME": "analysis",
        })
        self.assertEqual(
            config,
            {
                "host": "db.internal",
                "user": "sentiment",
                "password": "another-fixture",
                "database": "analysis",
            },
        )

    def test_retained_apps_use_environment_config_not_literal_password(self):
        for name in APP_FILES:
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn(
                "DatabaseHandler(**database_config_from_env())", source, name
            )
            self.assertIn(
                "from application_config import database_config_from_env", source, name
            )
            # A DB password assignment in these public scripts would bypass
            # the required environment-secret boundary.
            self.assertNotRegex(
                source,
                r"(?i)password\s*=\s*['\"][^'\"]+['\"]",
                name,
            )


if __name__ == "__main__":
    unittest.main()
