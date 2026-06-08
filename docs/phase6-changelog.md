# Phase 6 Changelog — GCP Platform Layer

**Project:** ERP AI — Delay Risk Prediction System  
**Owner:** Dr. Tatianna Gilliam  
**Date:** June 2026

## Overview

Phase 6 links ERP AI Delay Risk to two companion GCP portfolio projects that represent the enterprise infrastructure layers above the monolithic Cloud Run deployment. No breaking changes to inference endpoints in this repo.

## Added

### Documentation (`docs/`)

- `gcp-platform-roadmap.md` — Phase 6 (data platform) and Phase 7 (microservices) architecture map with repo links, feature alignment, and multi-cloud positioning
- `phase6-changelog.md` — This file

### README

- Phase 6 section with links to companion repos and key architecture documents

## Companion Projects (sibling repos)

| Project | GitHub | Layer |
|---------|--------|-------|
| GCP Enterprise Data Pipeline | [gcp-enterprise-data-pipeline](https://github.com/Tmgilliam/gcp-enterprise-data-pipeline) | BigQuery, Dataflow, Vertex AI Feature Store, ML pipelines |
| GCP Microservices + Apigee | [gcp-microservices-api-gateway](https://github.com/Tmgilliam/gcp-microservices-api-gateway) | Cloud Run microservices, Apigee API gateway |

## What Phase 6 Proves

- GCP fluency beyond "deployed a container" — data platform and API governance depth
- Platform-agnostic patterns with GCP production proof and Azure migration path already documented
- Clear separation: application layer (this repo) vs. data platform vs. service decomposition

## Unchanged

- All `/predict`, `/predict/batch`, `/health`, and `/monitoring/*` endpoints
- Phase 1–5 artifacts (Azure Bicep, Entra, APIM policies, drift monitoring)
- Default auth: `ENTRA_AUTH_ENABLED=false`
