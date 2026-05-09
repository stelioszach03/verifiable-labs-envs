#!/usr/bin/env bash
# _load_phase28_secrets.sh — Phase 28.D email + Slack alert provisioning.
#
# Stelios runs this once locally to load the email/slack secrets before
# 28.D testing against a real provider. Mirrors the Phase 23 R2 loader
# pattern. NEVER paste API keys into chat — this script reads them from
# silent prompts and exports to the current shell.
#
# Usage:
#   source ./_load_phase28_secrets.sh
#
# Exported vars:
#   VLABS_EMAIL_FROM            — From address for monitor alert emails
#                                 (e.g. "alerts@verifiable-labs.com").
#   VLABS_EMAIL_API_KEY         — Resend / SendGrid / SES API key.
#   VLABS_SLACK_WEBHOOK_DEFAULT — Optional default Slack webhook URL
#                                 used as a fallback when a monitor's
#                                 alert_channels list does not specify
#                                 a per-monitor Slack URL.
#
# Tests skip these — services/api/tests run in LOCAL_FAKE_EMAIL mode
# (writes .eml stubs to /tmp/vlabs-emails/<ts>.eml). Production deploys
# pull the real values from Fly secrets.

set -e

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this with bash:  source ./_load_phase28_secrets.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: this script must be sourced, not executed directly." >&2
  echo "Try:   source ./_load_phase28_secrets.sh" >&2
  exit 1
fi

read -srp "VLABS_EMAIL_FROM (e.g. alerts@verifiable-labs.com): " _vlabs_email_from
echo
read -srp "VLABS_EMAIL_API_KEY (Resend / SES key): " _vlabs_email_key
echo
read -srp "VLABS_SLACK_WEBHOOK_DEFAULT (blank to skip): " _vlabs_slack_default
echo

if [ -z "$_vlabs_email_from" ] || [ -z "$_vlabs_email_key" ]; then
  echo "ERROR: VLABS_EMAIL_FROM and VLABS_EMAIL_API_KEY are required." >&2
  unset _vlabs_email_from _vlabs_email_key _vlabs_slack_default
  return 1 2>/dev/null || exit 1
fi

export VLABS_EMAIL_FROM="$_vlabs_email_from"
export VLABS_EMAIL_API_KEY="$_vlabs_email_key"
if [ -n "$_vlabs_slack_default" ]; then
  export VLABS_SLACK_WEBHOOK_DEFAULT="$_vlabs_slack_default"
fi

# Wipe local copies; only the exported vars remain in memory.
unset _vlabs_email_from _vlabs_email_key _vlabs_slack_default

echo "Phase 28 secrets loaded:"
echo "  VLABS_EMAIL_FROM             length=${#VLABS_EMAIL_FROM}"
echo "  VLABS_EMAIL_API_KEY          length=${#VLABS_EMAIL_API_KEY}, tail=${VLABS_EMAIL_API_KEY: -8}"
if [ -n "${VLABS_SLACK_WEBHOOK_DEFAULT:-}" ]; then
  echo "  VLABS_SLACK_WEBHOOK_DEFAULT  length=${#VLABS_SLACK_WEBHOOK_DEFAULT}, tail=${VLABS_SLACK_WEBHOOK_DEFAULT: -8}"
else
  echo "  VLABS_SLACK_WEBHOOK_DEFAULT  (not set)"
fi
