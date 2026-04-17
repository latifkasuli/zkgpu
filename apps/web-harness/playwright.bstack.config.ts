/**
 * Playwright config for G.0.2 tail A/B on BrowserStack (direct-CDP).
 *
 * Separate from `playwright.config.ts` so local runs stay untouched.
 * The direct-CDP approach uses regular Playwright `projects`; a
 * `page` fixture in `tests/bstack-fixture.ts` dials the BrowserStack
 * WebSocket endpoint when a project's name contains `@browserstack`.
 *
 * Run:
 *   source ~/.browserstack/credentials
 *   npx playwright test --config playwright.bstack.config.ts
 */
import { defineConfig } from "@playwright/test";

const PROJECT_SUFFIX = "@browserstack";

// Per-project defaults for the launch flags the bstack-fixture
// forwards as caps. Chrome's `--enable-unsafe-webgpu` is belt-and-
// braces for non-allowlisted adapters (cloud GPUs frequently fall
// into this bucket). Firefox needs `dom.webgpu.enabled` on every
// version through 2026 — WebGPU isn't on by default yet on desktop
// Firefox builds BrowserStack serves.
const CHROME_WEBGPU_ARGS = ["--enable-unsafe-webgpu"];
const FIREFOX_WEBGPU_PREFS = {
  "dom.webgpu.enabled": true,
  // Firefox gates some WebGPU features behind Nightly; these are
  // harmless on releases that don't honor them.
  "dom.webgpu.workers.enabled": true,
  "gfx.webrender.all": true,
};

export default defineConfig({
  testDir: "./tests",
  testMatch: "g02_tail_ab.spec.ts",
  fullyParallel: false,
  workers: 2, // one BrowserStack session per Playwright worker
  retries: 0,
  timeout: 600_000, // cloud + tunnel + wasm build + 5-iter loops

  globalSetup: "./tests/bstack-global-setup.ts",
  globalTeardown: "./tests/bstack-global-teardown.ts",

  reporter: [
    ["list"],
    ["json", { outputFile: "test-results/g02-bstack.json" }],
  ],

  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
    // Navigation timeout — cloud round-trip to localhost via tunnel is slow.
    navigationTimeout: 120_000,
    actionTimeout: 60_000,
  },

  // Start the local Vite server. BrowserStack Local tunnel (brought
  // up by globalSetup) routes the cloud browser's requests back here.
  webServer: {
    command: "npm run build:wasm && npx vite --port 5173 --host 0.0.0.0",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 180_000,
  },

  projects: [
    // Windows 11 — BrowserStack Windows VMs don't expose WebGPU adapters
    // today (verified chrome@latest + playwright-chromium both fail
    // with "webgpu found no adapters"). Kept for future re-check.
    {
      name: `chrome@latest:Windows 11${PROJECT_SUFFIX}`,
      use: {
        browserName: "chromium",
        bstackBrowser: "chrome",
        bstackOs: "Windows",
        bstackOsVersion: "11",
        bstackArgs: CHROME_WEBGPU_ARGS,
      } as any,
    },
    {
      name: `playwright-firefox@latest:Windows 11${PROJECT_SUFFIX}`,
      use: {
        browserName: "firefox",
        bstackBrowser: "playwright-firefox",
        bstackOs: "Windows",
        bstackOsVersion: "11",
        bstackFirefoxPrefs: FIREFOX_WEBGPU_PREFS,
      } as any,
    },

    // macOS — Mac minis expose real Metal adapters to WebGPU.
    {
      name: `chrome@latest:OS X Sonoma${PROJECT_SUFFIX}`,
      use: {
        browserName: "chromium",
        bstackBrowser: "chrome",
        bstackOs: "OS X",
        bstackOsVersion: "Sonoma",
        bstackArgs: CHROME_WEBGPU_ARGS,
      } as any,
    },
    {
      name: `chrome@latest:OS X Sequoia${PROJECT_SUFFIX}`,
      use: {
        browserName: "chromium",
        bstackBrowser: "chrome",
        bstackOs: "OS X",
        bstackOsVersion: "Sequoia",
        bstackArgs: CHROME_WEBGPU_ARGS,
      } as any,
    },
    {
      name: `playwright-firefox@latest:OS X Sonoma${PROJECT_SUFFIX}`,
      use: {
        browserName: "firefox",
        bstackBrowser: "playwright-firefox",
        bstackOs: "OS X",
        bstackOsVersion: "Sonoma",
        bstackFirefoxPrefs: FIREFOX_WEBGPU_PREFS,
      } as any,
    },
  ],
});
