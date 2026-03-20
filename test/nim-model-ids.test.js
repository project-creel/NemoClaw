// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const {
  diffCatalogModelIds,
  getCatalogCloudModelIds,
  parseNvidiaModelIndex,
} = require("../bin/lib/nim-model-ids");

describe("nim model ids", () => {
  it("collects cloud-facing model ids from the catalog", () => {
    assert.ok(getCatalogCloudModelIds().includes("nvidia/nemotron-3-nano-30b-a3b"));
    assert.ok(getCatalogCloudModelIds().includes("z-ai/glm5"));
  });

  it("parses the NVIDIA model index response", () => {
    assert.deepEqual(
      parseNvidiaModelIndex({
        data: [{ id: "nvidia/nemotron-3-nano-30b-a3b" }, { id: "z-ai/glm5" }],
      }),
      ["nvidia/nemotron-3-nano-30b-a3b", "z-ai/glm5"]
    );
  });

  it("diffs catalog ids against the live model index ids", () => {
    const diff = diffCatalogModelIds(
      ["nvidia/nemotron-3-nano-30b-a3b", "z-ai/glm5", "extra/model"],
      ["nvidia/nemotron-3-nano-30b-a3b", "minimaxai/minimax-m2.5", "z-ai/glm5"]
    );

    assert.deepEqual(diff, {
      missingFromApi: ["minimaxai/minimax-m2.5"],
      extraInApi: ["extra/model"],
    });
  });
});
