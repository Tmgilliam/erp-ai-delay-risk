# Phase 5 Changelog — Entra ID + APIM Readiness

**Project:** ERP AI — Delay Risk Prediction System  
**Owner:** Dr. Tatianna Gilliam  
**Date:** June 2026

## Overview

Phase 5 builds all Entra ID and APIM configuration **without requiring a paid Azure subscription**. Paid services (Container Apps, APIM) remain deferred; enabling them is a flag flip and script run when ready.

## Added

### Entra ID (Free)

- `scripts/entra/register-apps.ps1` — creates API + Dashboard app registrations
- `config/entra/app-roles.json` — EXEC / OPS / ANALYST role documentation
- `app/auth/entra_jwt.py` — optional FastAPI JWT validation (`ENTRA_AUTH_ENABLED=false` default)
- `dashboard/auth.py` — unified auth: `env` | `entra` | `dual`
- `requirements-entra.txt` — PyJWT, MSAL

### APIM (Artifacts Only — Apply When Paid)

- `config/apim/policies/` — global, API, and operation policies
- `config/apim/openapi.json` — APIM import spec with Entra OAuth placeholders
- `scripts/azure/configure-apim.ps1` — `-ValidateOnly` (free) / `-Apply` (paid)

### Configuration

- `config/feature-flags.env.example` — documents all enable flags
- `docs/entra-apim-readiness.md` — master guide

## Changed

- `app/main.py` — protected routes accept optional Entra JWT dependency
- `dashboard/app.py` — uses `dashboard/auth.py`; passes Bearer token to API when Entra session active

## Defaults (Unchanged Behavior)

| Setting | Default | Effect |
|---------|---------|--------|
| `ENTRA_AUTH_ENABLED` | `false` | API works without tokens (GCP/local) |
| `AUTH_MODE` | `env` | Dashboard uses DASH_USER/DASH_PASS |
| `APIM_ENABLED` | `false` | Direct API access |

## Enable When Ready

```powershell
# Free — do now
.\scripts\entra\register-apps.ps1
.\scripts\azure\configure-apim.ps1 -ValidateOnly

# Paid — when subscription active
.\scripts\azure\deploy.ps1 -DeployApim
.\scripts\azure\configure-apim.ps1 -Apply -ResourceGroup ... -ApimName ... -ApiBackendUrl ...
```

## Local Bootstrap (Phase 5b)

- `scripts/local/preflight.ps1` — prerequisite check
- `scripts/local/start-dev.ps1` — one-command local stack (env auth)
- `scripts/local/start-entra-dev.ps1` — Entra mode (needs generated.env)
- `docs/entra-portal-setup.md` — manual Entra when Azure CLI unavailable
- `config/entra/generated.env.template` — fill-in template

## Next Steps

1. **Now (free):** `.\scripts\local\start-dev.ps1` OR manual Entra via `docs/entra-portal-setup.md`
2. **When CLI available:** `winget install Microsoft.AzureCLI` → `az login` → `register-apps.ps1`
3. **When subscription funded:** `deploy.ps1` → `configure-apim.ps1 -Apply`
