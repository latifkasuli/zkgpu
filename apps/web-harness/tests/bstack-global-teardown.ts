/**
 * Stops the BrowserStack Local tunnel started by
 * `bstack-global-setup.ts`. Referenced as `globalTeardown` in
 * `playwright.bstack.config.ts`.
 */
import type { Local } from "browserstack-local";
import { clearLocalIdentifier } from "./bstack-fixture";

export default async function globalTeardown(): Promise<void> {
  const handle = (globalThis as unknown as { __bsLocal?: Local }).__bsLocal;
  if (handle) {
    await new Promise<void>((resolve) => {
      handle.stop(() => resolve());
    });
    console.log(`[bstack-global-teardown] Local tunnel stopped`);
  }
  clearLocalIdentifier();
}
