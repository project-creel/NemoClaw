#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Generic bridge runner.
 *
 * Reads bridge definitions from nemoclaw-blueprint/bridges/<type>/*.yaml,
 * loads the corresponding adapter, and runs the message flow. Credentials
 * stay on the host — messages relay to the sandbox via OpenShell SSH.
 *
 * Usage:
 *   node scripts/bridge.js <name>          Run a specific bridge by name
 *   node scripts/bridge.js --list          List available bridges
 *
 * Env:
 *   NVIDIA_API_KEY   — required for inference
 *   SANDBOX_NAME     — sandbox name (default: nemoclaw)
 *   Platform-specific tokens (see bridge YAML for token_env)
 */

const fs = require("fs");
const path = require("path");
const { runAgentInSandbox, SANDBOX } = require("./bridge-core");

const yaml = require("js-yaml");

const BRIDGES_DIR = path.join(__dirname, "..", "nemoclaw-blueprint", "bridges");

// ── Load bridge configs ───────────────────────────────────────────

function loadBridgeConfigs() {
  const configs = [];
  if (!fs.existsSync(BRIDGES_DIR)) return configs;

  for (const typeDir of fs.readdirSync(BRIDGES_DIR, { withFileTypes: true })) {
    if (!typeDir.isDirectory()) continue;
    const typePath = path.join(BRIDGES_DIR, typeDir.name);

    for (const file of fs.readdirSync(typePath)) {
      if (!file.endsWith(".yaml") && !file.endsWith(".yml")) continue;
      const content = fs.readFileSync(path.join(typePath, file), "utf-8");
      const parsed = yaml.load(content);
      if (parsed.bridge) configs.push(parsed.bridge);
    }
  }

  return configs;
}

function findAdapter(config) {
  const adapterPath = path.join(__dirname, "adapters", config.type, `${config.adapter}.js`);
  if (!fs.existsSync(adapterPath)) {
    console.error(`Adapter not found: ${adapterPath}`);
    return null;
  }
  return require(adapterPath);
}

// ── Message flow engine ───────────────────────────────────────────

async function runBridge(config) {
  const tokenEnv = config.credentials.token_env;
  const token = process.env[tokenEnv];
  if (!token) {
    console.error(`${tokenEnv} required for ${config.name} bridge`);
    process.exit(1);
  }

  // Check extra required env vars (e.g., SLACK_APP_TOKEN)
  const extraEnvs = config.credentials.extra_env;
  if (Array.isArray(extraEnvs)) {
    for (const env of extraEnvs) {
      if (!process.env[env]) {
        console.error(`${env} required for ${config.name} bridge`);
        process.exit(1);
      }
    }
  }

  const createAdapter = findAdapter(config);
  if (!createAdapter) process.exit(1);

  const adapter = createAdapter(config);
  const prefix = config.messaging.session_prefix;
  const maxChunk = config.messaging.max_chunk_size;

  async function onMessage(msg) {
    console.log(`[${config.name}] [${msg.channelId}] ${msg.userName}: inbound (len=${msg.text.length})`);

    // Typing indicator
    await msg.sendTyping();
    const typingInterval = setInterval(() => msg.sendTyping(), 4000);

    try {
      const response = await runAgentInSandbox(msg.text, `${prefix}-${msg.channelId}`);
      clearInterval(typingInterval);
      console.log(`[${config.name}] [${msg.channelId}] agent: response (len=${response.length})`);

      // Chunk response per platform limit
      const chunks = [];
      for (let i = 0; i < response.length; i += maxChunk) {
        chunks.push(response.slice(i, i + maxChunk));
      }
      for (const chunk of chunks) {
        await msg.reply(chunk);
      }
    } catch (err) {
      clearInterval(typingInterval);
      const errorMsg = err && err.message ? err.message : String(err);
      await msg.reply(`Error: ${errorMsg}`).catch(() => {});
    }
  }

  const botName = await adapter.start(onMessage);

  console.log("");
  console.log("  ┌─────────────────────────────────────────────────────┐");
  console.log(`  │  NemoClaw ${(config.name.charAt(0).toUpperCase() + config.name.slice(1) + " Bridge                     ").slice(0, 41)}│`);
  console.log("  │                                                     │");
  console.log(`  │  Bot:      ${(String(botName) + "                              ").slice(0, 41)}│`);
  console.log("  │  Sandbox:  " + (SANDBOX + "                              ").slice(0, 40) + "│");
  console.log("  │                                                     │");
  console.log("  │  Messages are forwarded to the OpenClaw agent      │");
  console.log("  │  inside the sandbox. Run 'openshell term' in       │");
  console.log("  │  another terminal to monitor + approve egress.     │");
  console.log("  └─────────────────────────────────────────────────────┘");
  console.log("");
}

// ── CLI ───────────────────────────────────────────────────────────

const args = process.argv.slice(2);

if (args[0] === "--list") {
  const configs = loadBridgeConfigs();
  console.log("\nAvailable bridges:\n");
  for (const c of configs) {
    const token = process.env[c.credentials.token_env] ? "✓" : "✗";
    console.log(`  ${token} ${c.name.padEnd(12)} ${c.description}  (${c.credentials.token_env})`);
  }
  console.log("");
  process.exit(0);
}

if (!args[0]) {
  console.error("Usage: node scripts/bridge.js <name>");
  console.error("       node scripts/bridge.js --list");
  process.exit(1);
}

const configs = loadBridgeConfigs();
const config = configs.find((c) => c.name === args[0]);
if (!config) {
  console.error(`Unknown bridge: ${args[0]}`);
  console.error(`Available: ${configs.map((c) => c.name).join(", ")}`);
  process.exit(1);
}

runBridge(config);
