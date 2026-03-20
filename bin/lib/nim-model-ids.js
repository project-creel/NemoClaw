// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const nimImages = require("./nim-images.json");

function getCatalogCloudModelIds() {
  return nimImages.models.map((model) => model.name);
}

function parseNvidiaModelIndex(payload) {
  let parsed = payload;
  if (typeof payload === "string") {
    parsed = JSON.parse(payload);
  }

  const rows = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.data) ? parsed.data : [];
  return rows
    .map((row) => (typeof row?.id === "string" ? row.id.trim() : ""))
    .filter(Boolean);
}

function diffCatalogModelIds(apiModelIds, catalogModelIds = getCatalogCloudModelIds()) {
  const apiSet = new Set(apiModelIds);
  const catalogSet = new Set(catalogModelIds);

  return {
    missingFromApi: catalogModelIds.filter((id) => !apiSet.has(id)),
    extraInApi: apiModelIds.filter((id) => !catalogSet.has(id)),
  };
}

module.exports = {
  diffCatalogModelIds,
  getCatalogCloudModelIds,
  parseNvidiaModelIndex,
};
