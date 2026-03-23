// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import path from "node:path";

import {
  detectDockerHost,
  findColimaDockerSocket,
  getDockerSocketCandidates,
  hasNfTablesNatSupport,
  inferContainerRuntime,
  isJetson,
  isUnsupportedMacosRuntime,
  isWsl,
  needsIptablesLegacy,
  shouldPatchCoredns,
} from "../bin/lib/platform";

describe("platform helpers", () => {
  describe("isWsl", () => {
    it("detects WSL from environment", () => {
      expect(isWsl({
        platform: "linux",
        env: { WSL_DISTRO_NAME: "Ubuntu" },
        release: "6.6.87.2-microsoft-standard-WSL2",
      })).toBe(true);
    });

    it("does not treat macOS as WSL", () => {
      expect(isWsl({
        platform: "darwin",
        env: {},
        release: "24.6.0",
      })).toBe(false);
    });
  });

  describe("getDockerSocketCandidates", () => {
    it("returns macOS candidates in priority order", () => {
      const home = "/tmp/test-home";
      expect(getDockerSocketCandidates({ platform: "darwin", home })).toEqual([
        path.join(home, ".colima/default/docker.sock"),
        path.join(home, ".config/colima/default/docker.sock"),
        path.join(home, ".docker/run/docker.sock"),
      ]);
    });

    it("does not auto-detect sockets on Linux", () => {
      expect(getDockerSocketCandidates({ platform: "linux", home: "/tmp/test-home" })).toEqual([]);
    });
  });

  describe("findColimaDockerSocket", () => {
    it("finds the first available Colima socket", () => {
      const home = "/tmp/test-home";
      const sockets = new Set([path.join(home, ".config/colima/default/docker.sock")]);
      const existsSync = (socketPath) => sockets.has(socketPath);

      expect(findColimaDockerSocket({ home, existsSync })).toBe(path.join(home, ".config/colima/default/docker.sock"));
    });
  });

  describe("detectDockerHost", () => {
    it("respects an existing DOCKER_HOST", () => {
      expect(detectDockerHost({
        env: { DOCKER_HOST: "unix:///custom/docker.sock" },
        platform: "darwin",
        home: "/tmp/test-home",
        existsSync: () => false,
      })).toEqual({
        dockerHost: "unix:///custom/docker.sock",
        source: "env",
        socketPath: null,
      });
    });

    it("prefers Colima over Docker Desktop on macOS", () => {
      const home = "/tmp/test-home";
      const sockets = new Set([
        path.join(home, ".colima/default/docker.sock"),
        path.join(home, ".docker/run/docker.sock"),
      ]);
      const existsSync = (socketPath) => sockets.has(socketPath);

      expect(detectDockerHost({ env: {}, platform: "darwin", home, existsSync })).toEqual({
        dockerHost: `unix://${path.join(home, ".colima/default/docker.sock")}`,
        source: "socket",
        socketPath: path.join(home, ".colima/default/docker.sock"),
      });
    });

    it("detects Docker Desktop when Colima is absent", () => {
      const home = "/tmp/test-home";
      const socketPath = path.join(home, ".docker/run/docker.sock");
      const existsSync = (candidate) => candidate === socketPath;

      expect(detectDockerHost({ env: {}, platform: "darwin", home, existsSync })).toEqual({
        dockerHost: `unix://${socketPath}`,
        source: "socket",
        socketPath,
      });
    });

    it("returns null when no auto-detected socket is available", () => {
      expect(detectDockerHost({
        env: {},
        platform: "linux",
        home: "/tmp/test-home",
        existsSync: () => false,
      })).toBe(null);
    });
  });

  describe("inferContainerRuntime", () => {
    it("detects podman", () => {
      expect(inferContainerRuntime("podman version 5.4.1")).toBe("podman");
    });

    it("detects Docker Desktop", () => {
      expect(inferContainerRuntime("Docker Desktop 4.42.0 (190636)")).toBe("docker-desktop");
    });

    it("detects Colima", () => {
      expect(inferContainerRuntime("Server: Colima\n Docker Engine - Community")).toBe("colima");
    });
  });

  describe("isUnsupportedMacosRuntime", () => {
    it("flags podman on macOS", () => {
      expect(isUnsupportedMacosRuntime("podman", { platform: "darwin" })).toBe(true);
    });

    it("does not flag podman on Linux", () => {
      expect(isUnsupportedMacosRuntime("podman", { platform: "linux" })).toBe(false);
    });
  });

  describe("shouldPatchCoredns", () => {
    it("patches CoreDNS for Colima only", () => {
      expect(shouldPatchCoredns("colima")).toBe(true);
      expect(shouldPatchCoredns("docker-desktop")).toBe(false);
      expect(shouldPatchCoredns("docker")).toBe(false);
    });
  });

  describe("isJetson", () => {
    it("detects Jetson from kernel release string", () => {
      assert.equal(
        isJetson({
          platform: "linux",
          release: "5.15.185-tegra",
          deviceTreeCompatible: "",
          nvTegraRelease: "",
        }),
        true,
      );
    });

    it("detects Jetson from device tree compatible", () => {
      assert.equal(
        isJetson({
          platform: "linux",
          release: "5.15.0-generic",
          deviceTreeCompatible: "nvidia,p3768-0000+p3767-0005\0nvidia,jetson-orin-nano\0nvidia,tegra234\0",
          nvTegraRelease: "",
        }),
        true,
      );
    });

    it("detects Jetson from /etc/nv_tegra_release", () => {
      assert.equal(
        isJetson({
          platform: "linux",
          release: "5.15.0-generic",
          deviceTreeCompatible: "",
          nvTegraRelease: "# R36 (release), REVISION: 4.3",
        }),
        true,
      );
    });

    it("returns false for non-Jetson Linux", () => {
      assert.equal(
        isJetson({
          platform: "linux",
          release: "6.8.0-41-generic",
          deviceTreeCompatible: "",
          nvTegraRelease: "",
        }),
        false,
      );
    });

    it("returns false for macOS", () => {
      assert.equal(
        isJetson({
          platform: "darwin",
          release: "24.6.0",
          deviceTreeCompatible: "",
          nvTegraRelease: "",
        }),
        false,
      );
    });
  });

  describe("hasNfTablesNatSupport", () => {
    it("returns true when nft_chain_nat is loaded", () => {
      assert.equal(
        hasNfTablesNatSupport({
          platform: "linux",
          procModules: "nft_chain_nat 16384 1 - Live 0xffff\nnf_nat 49152 2 - Live 0xffff",
        }),
        true,
      );
    });

    it("returns false when nft_chain_nat is absent", () => {
      assert.equal(
        hasNfTablesNatSupport({
          platform: "linux",
          procModules: "nf_conntrack 163840 3 - Live 0xffff\nip_tables 32768 0 - Live 0xffff",
        }),
        false,
      );
    });

    it("returns true on non-Linux (not applicable)", () => {
      assert.equal(hasNfTablesNatSupport({ platform: "darwin" }), true);
    });
  });

  describe("needsIptablesLegacy", () => {
    it("returns true for Jetson without nf_tables NAT", () => {
      assert.equal(
        needsIptablesLegacy({
          platform: "linux",
          release: "5.15.185-tegra",
          deviceTreeCompatible: "",
          nvTegraRelease: "",
          procModules: "ip_tables 32768 0 - Live 0xffff",
        }),
        true,
      );
    });

    it("returns false for Jetson with nf_tables NAT", () => {
      assert.equal(
        needsIptablesLegacy({
          platform: "linux",
          release: "5.15.185-tegra",
          deviceTreeCompatible: "",
          nvTegraRelease: "",
          procModules: "nft_chain_nat 16384 1 - Live 0xffff",
        }),
        false,
      );
    });

    it("returns false for non-Jetson Linux", () => {
      assert.equal(
        needsIptablesLegacy({
          platform: "linux",
          release: "6.8.0-41-generic",
          deviceTreeCompatible: "",
          nvTegraRelease: "",
          procModules: "",
        }),
        false,
      );
    });

    it("returns false for macOS", () => {
      assert.equal(needsIptablesLegacy({ platform: "darwin" }), false);
    });
  });
});
