#!/usr/bin/env python3
"""Start the WC2026 Odds Engine web app.  Env: PORT (default 8000), WC_DB,
WC_AUTO_FETCH=1 to pull openfootball results every 6h."""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run("webapp.server:app",
                host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))
