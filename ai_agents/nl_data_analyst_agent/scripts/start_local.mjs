import { spawnSync, spawn } from "node:child_process";
import { resolve } from "node:path";

const cli = resolve("node_modules/eve/bin/eve.js");
const built = spawnSync(process.execPath, [cli, "build"], { stdio: "inherit" });
if (built.status !== 0) process.exit(built.status ?? 1);

// Local-only preview: Eve's localDev authenticator checks EVE_DEV.
const child = spawn(process.execPath, [cli, "start", "--host", "127.0.0.1", "--port", "3000"], {
  env: { ...process.env, EVE_DEV: "1" },
  stdio: "inherit",
});
child.on("exit", (code) => process.exit(code ?? 0));
process.on("SIGINT", () => child.kill("SIGINT"));
process.on("SIGTERM", () => child.kill("SIGTERM"));
