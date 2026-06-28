#!/bin/bash
set -e

# รันบน VPS ที่ติดตั้ง Vault แล้ว
# ใช้: VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=<root-token> bash setup.sh

if [ -z "$VAULT_ADDR" ] || [ -z "$VAULT_TOKEN" ]; then
  echo "ERROR: กรุณาตั้งค่า VAULT_ADDR และ VAULT_TOKEN ก่อนรัน script นี้"
  echo "ตัวอย่าง: VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=hvs.xxx bash setup.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating Jenkins Vault policy..."
vault policy write jenkins-ml-api "$SCRIPT_DIR/jenkins-policy.hcl"
echo "    Policy 'jenkins-ml-api' created."

echo "==> Creating Jenkins token (no expiry, orphan, read-only)..."
vault token create \
  -policy="jenkins-ml-api" \
  -display-name="jenkins-leaf-classification" \
  -no-default-policy \
  -orphan \
  -format=json > "$SCRIPT_DIR/jenkins-token.json"

JENKINS_TOKEN=$(cat "$SCRIPT_DIR/jenkins-token.json" | grep '"client_token"' | awk -F'"' '{print $4}')

echo ""
echo "============================================================"
echo "  Jenkins Vault Token (copy นี้ไปใส่ใน Jenkins Credentials)"
echo "============================================================"
echo "  $JENKINS_TOKEN"
echo "============================================================"
echo ""
echo "วิธีใส่ใน Jenkins:"
echo "  1. Manage Jenkins → Credentials → System → Global credentials"
echo "  2. Add Credentials"
echo "     Kind: Secret text"
echo "     Secret: <token ด้านบน>"
echo "     ID: vault-token"
echo ""
echo "จากนั้นลบไฟล์ token:"
echo "  rm $SCRIPT_DIR/jenkins-token.json"
