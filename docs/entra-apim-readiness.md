# Entra ID + APIM Readiness Guide

**Owner:** Dr. Tatianna Gilliam  
**Purpose:** Build everything now; enable paid services when subscription is active.

---

## What Costs Money vs. What's Free

| Component | Cost | Status in This Repo |
|-----------|------|---------------------|
| **Entra ID app registrations** | Free | `scripts/entra/register-apps.ps1` |
| **Entra JWT auth in API** | Free | `app/auth/entra_jwt.py` (disabled by default) |
| **Dashboard Entra login** | Free | `dashboard/auth.py` (`AUTH_MODE=entra`) |
| **APIM policy files** | Free (artifacts only) | `config/apim/policies/` |
| **OpenAPI spec for APIM** | Free | `config/apim/openapi.json` |
| **Azure Container Apps** | Paid (~$15–30/mo) | `scripts/azure/deploy.ps1` — defer |
| **APIM Developer tier** | Paid (~$50/mo) | `configure-apim.ps1 -Apply` — defer |

You can complete Entra registration and local auth testing **today** without an Azure subscription.

---

## Phase 1 - Entra ID (Do Now, Free)

> **No Azure CLI?** Use [entra-portal-setup.md](entra-portal-setup.md) for manual portal steps, then `config/entra/generated.env.template` → `generated.env`.

### 1. Register applications

```powershell
cd C:\Users\teemm\MTP-Projects\erp-ai-delay-risk
az login
.\scripts\entra\register-apps.ps1
```

This creates:
- **ERP AI Delay Risk API** — app roles `EXEC`, `OPS`, `ANALYST` + `predict` scope
- **ERP AI Delay Risk Dashboard** — delegated permission to call API

Output: `config/entra/generated.env` (gitignored)

### 2. Assign roles (Entra Portal)

1. **Entra ID** → **App registrations** → **ERP AI Delay Risk API**
2. **App roles** → Assign users/groups to EXEC, OPS, or ANALYST
3. **API permissions** on Dashboard app → **Grant admin consent**

### 3. Test locally (no paid Azure)

```powershell
pip install -r requirements-entra.txt

# Terminal 1 — API with Entra JWT enforcement
$env:ENTRA_AUTH_ENABLED = "true"
Get-Content config\entra\generated.env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
uvicorn app.main:app --port 8001

# Terminal 2 — Dashboard with Entra device code login
$env:AUTH_MODE = "entra"
$env:API_URL = "http://127.0.0.1:8001"
# load generated.env vars...
streamlit run dashboard/app.py
```

### 4. Auth modes

| `AUTH_MODE` | Behavior |
|-------------|----------|
| `env` | DASH_USER/DASH_PASS (default — portfolio demos) |
| `entra` | Microsoft device code sign-in |
| `dual` | User chooses env or Entra |

---

## Phase 2 — APIM Artifacts (Do Now, Free)

Validate policy files and OpenAPI without deploying APIM:

```powershell
.\scripts\azure\configure-apim.ps1 -ValidateOnly
```

### Policy files

| File | Purpose |
|------|---------|
| `config/apim/policies/global-policy.xml` | Rate limit, correlation ID |
| `config/apim/policies/api-inbound-policy.xml` | Backend routing + Entra JWT (commented) |
| `config/apim/policies/predict-operation-policy.xml` | Stricter limit on `/predict` |

### Enable Entra JWT at gateway (when APIM is live)

1. Run `register-apps.ps1` first (populates tenant/audience placeholders)
2. Edit `api-inbound-policy.xml` — **uncomment** the `<validate-jwt>` block
3. Run `configure-apim.ps1 -Apply` after APIM provisions

---

## Phase 3 — Paid Azure (When Subscription Active)

### Order of operations

```
1. deploy.ps1          → Container Apps, ACR, Blob, Storage (~$15–30/mo)
2. register-apps.ps1   → Entra (if not done) — FREE
3. deploy.ps1 -DeployApim  OR  enable APIM in Bicep (~$50/mo additional)
4. configure-apim.ps1 -Apply  → Import OpenAPI, apply policies
5. Set ENTRA_AUTH_ENABLED=true on API Container App
6. Set AUTH_MODE=entra on Dashboard Container App
7. Point ERP integrators to APIM gateway URL
```

### Feature flags

Copy `config/feature-flags.env.example` and set per environment:

```env
ENTRA_AUTH_ENABLED=false   # true when Entra tested
APIM_ENABLED=false         # true when APIM provisioned
AUTH_MODE=env              # entra when SSO ready
```

---

## Architecture When Fully Enabled

```
ERP Client / Dashboard
        │
        ▼
  Azure API Management
  ├── validate-jwt (Entra)
  ├── rate-limit
  └── /v1/predict → backend
        │
        ▼
  Container Apps API
  ├── ENTRA_AUTH_ENABLED=true (defense in depth)
  ├── Blob model load (managed identity)
  └── drift + scoring history
```

**Defense in depth:** APIM validates JWT at the gateway; API can independently enforce Entra auth. Disable API-level auth if APIM is the sole enforcement point.

---

## Interview Talking Points

- *"Entra app registrations and RBAC roles are configured — EXEC, OPS, ANALYST map to how operations teams actually work."*
- *"APIM policies are written and validated; JWT validation is one uncomment away when the gateway is provisioned."*
- *"I deliberately separated free identity/governance prep from paid compute so the portfolio demonstrates enterprise thinking without requiring subscription spend upfront."*

**Power phrase:** *"I didn't model delay risk from the outside — I spent years inside the system that generates it."*

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `register-apps.ps1` fails | Ensure `az login` and Global Administrator or Application Administrator role |
| Device code login hangs | Grant admin consent on dashboard → API scope |
| API returns 401 with Entra enabled | Check `ENTRA_API_AUDIENCE=api://{client-id}` matches token audience |
| APIM validate-jwt fails | Uncomment policy block; verify tenant ID and audience placeholders replaced |

---

## Files Reference

| Path | Purpose |
|------|---------|
| `scripts/entra/register-apps.ps1` | Free Entra app registration |
| `scripts/azure/configure-apim.ps1` | Validate / apply APIM config |
| `app/auth/entra_jwt.py` | Optional API JWT validation |
| `dashboard/auth.py` | Unified dashboard auth |
| `config/apim/` | Policy XML + OpenAPI |
| `config/entra/app-roles.json` | Role/scope documentation |
| `config/feature-flags.env.example` | Environment toggles |
