"""Runtime configuration for the retained sentiment-analysis application."""
from __future__ import annotations

import os
from typing import Mapping


def database_config_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return MySQL connection settings without embedding credentials in source.

    Host/user/database retain the historical local-development defaults. The
    password is deliberately required from the environment because this public
    repository must not ship an authentication secret.
    """
    env = os.environ if environ is None else environ
    password = str(env.get("SENTIMENT_DB_PASSWORD", ""))
    if not password:
        raise RuntimeError(
            "database persistence requires SENTIMENT_DB_PASSWORD in the environment"
        )
    return {
        "host": str(env.get("SENTIMENT_DB_HOST", "localhost")),
        "user": str(env.get("SENTIMENT_DB_USER", "root")),
        "password": password,
        "database": str(env.get("SENTIMENT_DB_NAME", "SentimentAnalysis")),
    }
