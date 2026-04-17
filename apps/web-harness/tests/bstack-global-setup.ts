/**
 * Starts BrowserStack Local tunnel. Referenced as `globalSetup`
 * in `playwright.bstack.config.ts`. The tunnel lets remote cloud
 * browsers reach localhost:5173 (our Vite dev server). A per-run
 * identifier isolates concurrent runs.
 */
import { Local } from "browserstack-local";
import { generateLocalIdentifier, writeLocalIdentifier } from "./bstack-fixture";

export default async function globalSetup(): Promise<void> {
  const accessKey = process.env.BROWSERSTACK_ACCESS_KEY;
  if (!accessKey) {
    throw new Error(
      "BROWSERSTACK_ACCESS_KEY not set — cannot start BrowserStack Local tunnel",
    );
  }

  // Generate and persist the identifier so worker processes pick up
  // the same value when they construct per-session caps.
  const identifier = generateLocalIdentifier();
  writeLocalIdentifier(identifier);

  const bsLocal = new Local();
  await new Promise<void>((resolve, reject) => {
    bsLocal.start(
      {
        key: accessKey,
        localIdentifier: identifier,
        forcelocal: "true",
      },
      (err?: Error) => {
        if (err) reject(err);
        else resolve();
      },
    );
  });

  console.log(
    `[bstack-global-setup] Local tunnel up; identifier=${identifier}`,
  );

  // Park on globalThis so the teardown hook can reach the same instance.
  (globalThis as unknown as { __bsLocal: Local }).__bsLocal = bsLocal;
}
