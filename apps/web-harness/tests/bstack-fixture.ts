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
  browser_version?: string;
  os?: string;
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

  // Real-device caps (iOS iPhone/iPad). When `device` is set, the
  // session is scheduled on a real mobile device rather than a desktop
  // VM; `os_version` becomes the iOS version (e.g. "18.0").
  device?: string;
  realMobile?: "true" | "false";

  // Browser-launch tunables passed through to the remote browser.
  // BrowserStack's CDP ingress forwards `args` to Chromium-family
  // command lines and `firefox_user_prefs` to Firefox's about:config.
  args?: string[];
  firefox_user_prefs?: Record<string, string | number | boolean>;
}

export interface BstackProjectUse {
  bstackOs?: string;
  bstackOsVersion?: string;
  bstackBrowser?: string;
  bstackBrowserVersion?: string;
  // Real-mobile-device targeting. When `bstackDevice` is set the
  // session runs on a real iPhone/iPad rather than a desktop VM;
  // `bstackOs` should be "ios" and `bstackOsVersion` the iOS release
  // ("18.0", "18.2", "18.6"). `bstackOs` / `bstackOsVersion` are
  // omitted entirely when `bstackDevice` is present and BrowserStack
  // uses the device-version mapping implicit in the name.
  bstackDevice?: string;
  bstackRealMobile?: boolean;
  // Chromium-family command-line args (e.g. "--enable-unsafe-webgpu").
  bstackArgs?: string[];
  // Firefox about:config prefs (e.g. { "dom.webgpu.enabled": true }).
  bstackFirefoxPrefs?: Record<string, string | number | boolean>;
}

type DerivedCaps = Pick<
  BrowserStackCaps,
  "browser" | "browser_version" | "os" | "os_version" | "device" | "realMobile"
>;

function capsFromProject(
  name: string,
  use: BstackProjectUse,
): DerivedCaps {
  // Real iOS device: BrowserStack uses `device` + `os_version` and
  // omits `os` (the device name determines the platform).
  if (use.bstackDevice) {
    return {
      browser: use.bstackBrowser ?? "playwright-webkit",
      os_version: use.bstackOsVersion ?? "",
      device: use.bstackDevice,
      realMobile: (use.bstackRealMobile ?? true) ? "true" : "false",
    };
  }
  // Desktop VM: explicit bstack* fields preferred over name-parsing.
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
      ...(projectUse.bstackArgs ? { args: projectUse.bstackArgs } : {}),
      ...(projectUse.bstackFirefoxPrefs
        ? { firefox_user_prefs: projectUse.bstackFirefoxPrefs }
        : {}),
    };

    // All BrowserStack Playwright sessions dial through the chromium
    // playwright entry point — it's the WebSocket ingress, not the
    // engine selection (which is encoded in caps.browser).
    const wsEndpoint =
      `wss://cdp.browserstack.com/playwright?caps=` +
      encodeURIComponent(JSON.stringify(caps));

    // iOS runs identify by `device`; desktop runs by `os`. Log whichever
    // applies so the harness output stays readable across platforms.
    const target = caps.device
      ? `${caps.device} iOS ${caps.os_version}`
      : `${caps.os ?? "?"} ${caps.os_version}`;
    console.log(
      `[bstack-fixture] Connecting to BrowserStack (`
      + `${caps.browser}@${caps.browser_version ?? "latest"} on ${target})`,
    );
    const browser = await playwright.chromium.connect({
      wsEndpoint,
      timeout: 120_000,
    });
    console.log(`[bstack-fixture] Connected — creating page`);
    // BrowserStack's CDP variant expects newPage() directly on the
    // Browser; it doesn't expose the regular newContext surface.
    //
    // Real iOS/Android devices can't resolve `localhost` through the
    // BrowserStack Local tunnel — the tunnel proxies hostname
    // `bs-local.com` (127.0.0.1 in DNS on the device side) back to
    // the dev machine's loopback. Swap the baseURL hostname when a
    // real device cap is in play; desktop VMs keep plain `localhost`.
    const effectiveBaseURL = caps.device && baseURL
      ? baseURL.replace(/^(https?:\/\/)(localhost|127\.0\.0\.1)/, "$1bs-local.com")
      : baseURL;
    if (caps.device && effectiveBaseURL !== baseURL) {
      console.log(`[bstack-fixture] iOS/device — rewriting baseURL to ${effectiveBaseURL}`);
    }
    const remotePage = await browser.newPage({ baseURL: effectiveBaseURL });
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
