#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Init script for hosted NemoClaw deployments.
# Runs as a K8s init container (or pre-start hook) to fetch config and
# secrets from external sources and write them to the shared volume
# that nemoclaw-start.sh reads in hosted mode.
#
# Supports two config sources:
#   1. S3 bucket (set CONFIG_BUCKET + TENANT_ID + CLAW_ID)
#   2. Local volume mount (set NEMOCLAW_CONFIG_PATH directly)
#
# Required env (S3 mode):
#   CONFIG_BUCKET   S3 bucket name (e.g., creel-agent-configs-519570321834)
#   TENANT_ID       Tenant identifier
#   CLAW_ID         Claw instance identifier
#   AWS_REGION      AWS region (default: us-east-1)
#
# Required env (volume mode):
#   NEMOCLAW_CONFIG_PATH   Path to pre-mounted config.json
#   NEMOCLAW_SECRETS_PATH  Path to pre-mounted secrets env file (optional)
#
# Optional env:
#   LITELLM_UPSTREAM       LLM gateway URL (injected into config)
#   LITELLM_API_KEY        LLM gateway API key (injected into secrets)
#   CLAW_SIZE              T-shirt size (S/M/L/XL) — written to env for entrypoint
#   OTEL_ENABLED           Enable OTEL diagnostics (default: true)
#
# Output (shared volume at /opt/nemoclaw/config):
#   config.json            OpenClaw gateway configuration
#   .secrets.env           KEY=VALUE secrets file (chmod 600)

set -euo pipefail

OUTPUT_DIR="${NEMOCLAW_INIT_OUTPUT:-/opt/nemoclaw/config}"
AWS_REGION="${AWS_REGION:-us-east-1}"

mkdir -p "$OUTPUT_DIR"

# ── S3 fetch mode ──────────────────────────────────────────────────
fetch_from_s3() {
  local bucket="$1" tenant="$2" claw="$3"
  local config_key="${tenant}/${claw}/config.json"
  local secrets_key="${tenant}/${claw}/secrets.json"

  echo "[init] Fetching config from s3://${bucket}/${config_key}" >&2
  aws s3 cp "s3://${bucket}/${config_key}" "${OUTPUT_DIR}/config.json" \
    --region "$AWS_REGION" --quiet

  # Patch config for hosted mode
  patch_config "${OUTPUT_DIR}/config.json"

  # Fetch secrets (optional — may not exist for all claws)
  if aws s3 cp "s3://${bucket}/${secrets_key}" /tmp/secrets.json \
    --region "$AWS_REGION" --quiet 2>/dev/null; then
    echo "[init] Secrets fetched from S3" >&2
    convert_secrets_json /tmp/secrets.json "${OUTPUT_DIR}/.secrets.env"
    rm -f /tmp/secrets.json
  else
    echo "[init] No secrets.json found in S3 — skipping" >&2
  fi

  # Sync skills if they exist
  local skills_prefix="${tenant}/${claw}/skills/"
  if aws s3 ls "s3://${bucket}/${skills_prefix}" --region "$AWS_REGION" >/dev/null 2>&1; then
    mkdir -p "${OUTPUT_DIR}/skills"
    aws s3 sync "s3://${bucket}/${skills_prefix}" "${OUTPUT_DIR}/skills/" \
      --region "$AWS_REGION" --quiet
    echo "[init] Skills synced from S3" >&2
  fi
}

# ── Config patching ────────────────────────────────────────────────
# Adds hosted-mode gateway settings to the externally-provided config.
patch_config() {
  local config_file="$1"

  python3 -c "
import json, os, secrets

config = json.load(open('${config_file}'))

# Ensure gateway section exists with hosted-mode defaults
gw = config.setdefault('gateway', {})
gw['mode'] = 'local'
cui = gw.setdefault('controlUi', {})
cui['dangerouslyAllowHostHeaderOriginFallback'] = True
cui['allowInsecureAuth'] = True

# Auth token — generate if not present
auth = gw.setdefault('auth', {})
if not auth.get('token'):
    auth['token'] = secrets.token_hex(32)

# Trusted proxies for K8s pod networking
gw.setdefault('trustedProxies', ['127.0.0.1', '::1'])

# Inject LLM gateway if LITELLM_UPSTREAM is set
litellm_url = os.environ.get('LITELLM_UPSTREAM', '')
if litellm_url:
    providers = config.setdefault('models', {}).setdefault('providers', {})
    providers.setdefault('hosted-gateway', {
        'baseUrl': litellm_url,
        'apiKey': 'injected-at-runtime',
        'api': 'openai-completions',
    })

# OTEL diagnostics
if os.environ.get('OTEL_ENABLED', 'true').lower() == 'true':
    diag = config.setdefault('diagnostics', {})
    diag['enabled'] = True

json.dump(config, open('${config_file}', 'w'), indent=2)
"
  echo "[init] Config patched for hosted mode" >&2
}

# ── Secrets conversion ─────────────────────────────────────────────
# Converts secrets.json ({"KEY": "value"}) to .secrets.env (KEY=value).
convert_secrets_json() {
  local src="$1" dst="$2"

  python3 -c "
import json
secrets = json.load(open('${src}'))
with open('${dst}', 'w') as f:
    for k, v in secrets.items():
        # Skip complex values (e.g., nested objects like whatsapp auth)
        if isinstance(v, str):
            f.write(f'{k}={v}\n')
"
  chmod 600 "$dst"
  echo "[init] Secrets written to ${dst}" >&2
}

# ── Inject runtime env vars into secrets ───────────────────────────
inject_runtime_secrets() {
  local secrets_file="${OUTPUT_DIR}/.secrets.env"
  touch "$secrets_file"
  chmod 600 "$secrets_file"

  # LiteLLM credentials
  if [ -n "${LITELLM_API_KEY:-}" ]; then
    echo "LITELLM_API_KEY=${LITELLM_API_KEY}" >>"$secrets_file"
  fi
  if [ -n "${LITELLM_UPSTREAM:-}" ]; then
    echo "LITELLM_UPSTREAM=${LITELLM_UPSTREAM}" >>"$secrets_file"
  fi

  # Claw sizing
  if [ -n "${CLAW_SIZE:-}" ]; then
    echo "CLAW_SIZE=${CLAW_SIZE}" >>"$secrets_file"
  fi
}

# ── Main ───────────────────────────────────────────────────────────

if [ -n "${CONFIG_BUCKET:-}" ] && [ -n "${TENANT_ID:-}" ] && [ -n "${CLAW_ID:-}" ]; then
  # S3 mode
  fetch_from_s3 "$CONFIG_BUCKET" "$TENANT_ID" "$CLAW_ID"
  inject_runtime_secrets
  echo "[init] S3 init complete — config at ${OUTPUT_DIR}" >&2

elif [ -n "${NEMOCLAW_CONFIG_PATH:-}" ]; then
  # Volume mode — config already mounted, just patch it
  cp "$NEMOCLAW_CONFIG_PATH" "${OUTPUT_DIR}/config.json"
  patch_config "${OUTPUT_DIR}/config.json"

  if [ -n "${NEMOCLAW_SECRETS_PATH:-}" ] && [ -f "${NEMOCLAW_SECRETS_PATH}" ]; then
    cp "$NEMOCLAW_SECRETS_PATH" "${OUTPUT_DIR}/.secrets.env"
    chmod 600 "${OUTPUT_DIR}/.secrets.env"
  fi

  inject_runtime_secrets
  echo "[init] Volume init complete — config at ${OUTPUT_DIR}" >&2

else
  echo "[init] ERROR: Set CONFIG_BUCKET+TENANT_ID+CLAW_ID (S3) or NEMOCLAW_CONFIG_PATH (volume)" >&2
  exit 1
fi
