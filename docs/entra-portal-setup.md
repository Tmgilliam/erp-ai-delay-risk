# Entra ID Portal Setup (Manual)

Use this guide when Azure CLI (`az`) is not installed. App registrations are **free** — no Azure subscription required. A Microsoft 365 developer tenant or any Entra ID tenant works.

---

## Prerequisites

- Access to [Microsoft Entra admin center](https://entra.microsoft.com)
- **Application Administrator** or **Global Administrator** role

---

## Step 1 — Register the API App

1. **Entra ID** → **App registrations** → **New registration**
2. Name: `ERP AI Delay Risk API`
3. Supported account types: **Single tenant**
4. Register (no redirect URI needed for API)
5. Copy **Application (client) ID** → save as `ENTRA_API_CLIENT_ID`
6. Copy **Directory (tenant) ID** → save as `ENTRA_TENANT_ID`

### Expose API scope

1. **Expose an API** → Set Application ID URI: `api://{ENTRA_API_CLIENT_ID}`
2. **Add a scope**:
   - Scope name: `predict`
   - Admin consent display name: `Predict delay risk`
   - Admin consent description: `Score ERP orders for shipment delay risk`
   - State: Enabled
3. Copy the scope full name → `api://{client-id}/predict` = `ENTRA_API_SCOPE`

### Add app roles

1. **App roles** → **Create app role** (repeat for each):

| Display name | Allowed member type | Value | Description |
|--------------|---------------------|-------|-------------|
| EXEC | Users/Groups | EXEC | Executive dashboard access |
| OPS | Users/Groups | OPS | Operations scoring workflows |
| ANALYST | Users/Groups | ANALYST | Read-only monitoring access |

---

## Step 2 — Register the Dashboard App

1. **New registration**
2. Name: `ERP AI Delay Risk Dashboard`
3. Supported account types: **Single tenant**
4. Redirect URI: **Public client/native** → `http://localhost:8501`
5. Also add: `https://login.microsoftonline.com/common/oauth2/nativeclient`
6. Copy **Application (client) ID** → `ENTRA_DASHBOARD_CLIENT_ID`

### API permissions

1. **API permissions** → **Add a permission** → **My APIs**
2. Select **ERP AI Delay Risk API**
3. Delegated permissions → check `predict`
4. **Grant admin consent for [tenant]**

---

## Step 3 — Assign Users to Roles

1. **Enterprise applications** → **ERP AI Delay Risk API**
2. **Users and groups** → **Add user/group**
3. Assign yourself (or test users) to **EXEC**, **OPS**, or **ANALYST**

---

## Step 4 — Create generated.env

```powershell
cd C:\Users\teemm\MTP-Projects\erp-ai-delay-risk
Copy-Item config\entra\generated.env.template config\entra\generated.env
# Edit generated.env with your client IDs and tenant ID
```

Example:

```env
ENTRA_TENANT_ID=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
ENTRA_API_CLIENT_ID=11111111-2222-3333-4444-555555555555
ENTRA_API_AUDIENCE=api://11111111-2222-3333-4444-555555555555
ENTRA_DASHBOARD_CLIENT_ID=66666666-7777-8888-9999-000000000000
ENTRA_API_SCOPE=api://11111111-2222-3333-4444-555555555555/predict
ENTRA_AUTH_ENABLED=false
AUTH_MODE=env
```

---

## Step 5 — Test Locally

```powershell
pip install -r requirements-entra.txt

# Entra auth test
.\scripts\local\start-entra-dev.ps1
```

1. Open dashboard → **Sign in with Microsoft** (device code flow)
2. Complete login in browser
3. Score an order — dashboard passes Bearer token to API

---

## Step 6 — Enable APIM JWT (When Paid)

1. Edit `config/apim/policies/api-inbound-policy.xml`
2. Uncomment the `<validate-jwt>` block
3. Run `.\scripts\azure\configure-apim.ps1 -Apply ...`

Placeholders `{{ENTRA_TENANT_ID}}` and `{{ENTRA_API_AUDIENCE}}` are auto-filled from `generated.env`.

---

## Install Azure CLI (Optional — Automates Steps 1–4)

```powershell
winget install Microsoft.AzureCLI
az login
.\scripts\entra\register-apps.ps1
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Admin consent button greyed out | Need Application Administrator role |
| Device code login fails | Grant admin consent on dashboard API permission |
| API returns 401 | Set `ENTRA_API_AUDIENCE=api://{api-client-id}` exactly |
| No roles in token | Assign app role under Enterprise applications, not app registration |
