#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  diffCatalogModelIds,
  getCatalogCloudModelIds,
  parseNvidiaModelIndex,
} = require("../bin/lib/nim-model-ids.js");

const endpoint = process.env.NIM_MODEL_INDEX_URL || "https://integrate.api.nvidia.com/v1/models";
const apiKey = process.env.NVIDIA_API_KEY || "";

const headers = {
  Accept: "application/json",
};
if (apiKey) {
  headers.Authorization = `Bearer ${apiKey}`;
}

const response = await fetch(endpoint, { headers });
if (!response.ok) {
  console.error(`Failed to fetch ${endpoint}: ${response.status} ${response.statusText}`);
  process.exit(1);
}

const body = await response.text();
const apiModelIds = parseNvidiaModelIndex(body);
const diff = diffCatalogModelIds(apiModelIds, getCatalogCloudModelIds());

console.log(`Catalog ids: ${getCatalogCloudModelIds().length}`);
console.log(`API ids: ${apiModelIds.length}`);

if (diff.missingFromApi.length > 0) {
  console.log("");
  console.log("Missing from NVIDIA model API:");
  for (const id of diff.missingFromApi) {
    console.log(`- ${id}`);
  }
}

if (diff.extraInApi.length > 0) {
  console.log("");
  console.log("Additional ids in NVIDIA model API:");
  for (const id of diff.extraInApi) {
    console.log(`- ${id}`);
  }
}

if (diff.missingFromApi.length > 0) {
  process.exit(2);
}
