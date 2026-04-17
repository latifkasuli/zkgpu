/**
 * BrowserStack direct-CDP fixture for Playwright tests.
 *
 * When a test's project name contains `@browserstack`, the `page`
 * fixture dials `wss://cdp.browserstack.com/playwright` with caps
 * parsed from the project name, instead of using the locally-launched
 * browser. Pattern adapted from BrowserStack's official example repo
 * (github.com/BrowserStackCE/browserstack-examples-playwright).
 *
 * The project name format is:
 *   `<browser>@<version>:<os> <os_version>@browserstack[-<suffix>]`
 * For example:
 *   `playwright-chromium@latest:Windows 11@browserstack`
 *   `playwright-firefox@latest:Windows 11@browserstack`
 *   `chrome@latest:Windows 11@browserstack`
 *
 * BrowserStack Local tunneling is coordinated by globalSetup /
 * globalTeardown in `playwright.bstack.config.ts`.
 */
import { test as base, expect } from "@playwright/test";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

interface BrowserStackCaps {
  browser: string;
  browser_version: string;
  os: string;
  os_version: string;
  name: string;
  build: string;
  project?: string;
  "browserstack.username": string;
  "browserstack.accessKey": string;
  // Per BrowserStack docs, boolean caps are string-valued.
  "browserstack.local": "true" | "false";
  "browserstack.localIdentifier"?: string;
  "browserstack.networkLogs"?: "true" | "false";
  "browserstack.consoleLogs"?: string;
  "client.playwrightVersion": string;
}

export interface BstackProjectUse {
  bstackOs?: string;
  bstackOsVersion?: string;
  bstackBrowser?: string;
  bstackBrowserVersion?: string;
}

function capsFromProject(
  name: string,
  use: BstackProjectUse,
): Omit<BrowserStackCaps, "name" | "build" | "browserstack.username" | "browserstack.accessKey" | "browserstack.local" | "client.playwrightVersion"> {
  // If the project's `use` block supplies bstack* fields, prefer
  // them — unambiguous. Otherwise fall back to a best-effort parse
  // of the project name.
  if (use.bstackOs && use.bstackBrowser) {
    return {
      browser: use.bstackBrowser,
      browser_version: use.bstackBrowserVersion ?? "latest",
      os: use.bstackOs,
      os_version: use.bstackOsVersion ?? "",
    };
  }
  const combo = name.split(/@browserstack/)[0];
  const [browserPart, osPart = "Windows 11"] = combo.split(/:/);
  const [browser, browser_version] = browserPart.split(/@/);
  const osTokens = osPart.trim().split(/\s+/);
  const os = osTokens.shift() ?? "Windows";
  const os_version = osTokens.join(" ");
  return {
    browser: browser || "chrome",
    browser_version: browser_version || "latest",
    os,
    os_version,
  };
}

function getPlaywrightVersion(): string {
  // BrowserStack supports specific Playwright versions server-side.
  // Allow an override for when the installed client is newer than
  // what BrowserStack advertises — a supported string lets the CDP
  // session establish even if the local client is a later release.
  if (process.env.BROWSERSTACK_PLAYWRIGHT_VERSION) {
    return process.env.BROWSERSTACK_PLAYWRIGHT_VERSION;
  }
  try {
    const pkg = require("@playwright/test/package.json");
    return pkg.version as string;
  } catch {
    return "1.57.0";
  }
}

export const BSTACK_PROJECT_MARKER = "@browserstack";

// Shared across all Playwright worker processes via a file written
// by globalSetup. The identifier must match on both sides of the
// BrowserStack Local tunnel — the Local CLI registers it at tunnel
// start, and the cap-caller quotes it per-session.
const IDENTIFIER_FILE = path.join(
  process.cwd(),
  ".browserstack-local-identifier",
);

export function generateLocalIdentifier(): string {
  return (
    process.env.BROWSERSTACK_LOCAL_IDENTIFIER
    || `zkgpu-${crypto.randomBytes(4).toString("hex")}`
  );
}

export function writeLocalIdentifier(id: string): void {
  fs.writeFileSync(IDENTIFIER_FILE, id, "utf8");
}

export function readLocalIdentifier(): string {
  if (process.env.BROWSERSTACK_LOCAL_IDENTIFIER) {
    return process.env.BROWSERSTACK_LOCAL_IDENTIFIER;
  }
  if (!fs.existsSync(IDENTIFIER_FILE)) {
    throw new Error(
      `BrowserStack local identifier file missing (${IDENTIFIER_FILE}); `
        + "globalSetup must run before worker processes",
    );
  }
  return fs.readFileSync(IDENTIFIER_FILE, "utf8").trim();
}

export function clearLocalIdentifier(): void {
  try {
    fs.unlinkSync(IDENTIFIER_FILE);
  } catch {
    // Best-effort.
  }
}

export const test = base.extend({
  page: async ({ page, playwright, baseURL }, use, testInfo) => {
    if (!testInfo.project.name.includes(BSTACK_PROJECT_MARKER)) {
      await use(page);
      return;
    }

    // Close the locally-launched page — we're going remote.
    await page.close();

    const username = process.env.BROWSERSTACK_USERNAME;
    const accessKey = process.env.BROWSERSTACK_ACCESS_KEY;
    if (!username || !accessKey) {
      throw new Error(
        "BrowserStack project selected but BROWSERSTACK_USERNAME / BROWSERSTACK_ACCESS_KEY not set",
      );
    }

    const projectUse = (testInfo.project.use ?? {}) as BstackProjectUse;
    const parsed = capsFromProject(testInfo.project.name, projectUse);
    const caps: BrowserStackCaps = {
      ...parsed,
      name: `${testInfo.title} [${testInfo.project.name}]`,
      build: process.env.BROWSERSTACK_BUILD_NAME || "g02-browser-tail-ab",
      project: "zkgpu",
      "browserstack.username": username,
      "browserstack.accessKey": accessKey,
      "browserstack.local": "true",
      "browserstack.localIdentifier": readLocalIdentifier(),
      "browserstack.networkLogs": "false",
      "browserstack.consoleLogs": "errors",
      "client.playwrightVersion": getPlaywrightVersion(),
    };

    // All BrowserStack Playwright sessions dial through the chromium
    // playwright entry point — it's the WebSocket ingress, not the
    // engine selection (which is encoded in caps.browser).
    const wsEndpoint =
      `wss://cdp.browserstack.com/playwright?caps=` +
      encodeURIComponent(JSON.stringify(caps));

    console.log(
      `[bstack-fixture] Connecting to BrowserStack (`
      + `${caps.browser}@${caps.browser_version} on ${caps.os} ${caps.os_version})`,
    );
    const browser = await playwright.chromium.connect({
      wsEndpoint,
      timeout: 120_000,
    });
    console.log(`[bstack-fixture] Connected — creating page`);
    // BrowserStack's CDP variant expects newPage() directly on the
    // Browser; it doesn't expose the regular newContext surface.
    const remotePage = await browser.newPage({ baseURL });
    console.log(`[bstack-fixture] Page ready — handing to test`);

    try {
      await use(remotePage);
    } finally {
      // Report test status back to BrowserStack so the dashboard
      // shows the right pass/fail.
      const payload = {
        action: "setSessionStatus",
        arguments: {
          status: testInfo.status === "passed" ? "passed" : "failed",
          reason: testInfo.error?.message ?? "",
        },
      };
      try {
        await remotePage.evaluate(() => {}, `browserstack_executor: ${JSON.stringify(payload)}`);
      } catch {
        // Best-effort; session may already be in teardown.
      }
      await remotePage.close().catch(() => {});
      await browser.close().catch(() => {});
    }
  },
});

export { expect };
