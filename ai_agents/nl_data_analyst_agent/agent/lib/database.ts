import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const root = resolve(process.cwd());
const registryPath = join(root, "data", "connections.json");
const idPattern = /^(demo|[0-9a-f]{32})$/;

export function openDatabase(databaseId: string): DatabaseSync {
  if (!idPattern.test(databaseId)) throw new Error("Invalid database ID.");
  const registry = JSON.parse(readFileSync(registryPath, "utf8")) as Record<string, string>;
  const path = registry[databaseId];
  if (!path) throw new Error("Database is not connected.");
  const db = new DatabaseSync(path, { readOnly: true });
  db.exec("PRAGMA query_only = ON");
  return db;
}

export function validateSql(sql: string): string {
  const cleaned = sql.trim().replace(/;\s*$/, "");
  if (!/^select\b/i.test(cleaned) || cleaned.includes(";")) {
    throw new Error("Exactly one SELECT statement is allowed.");
  }
  if (/\b(attach|detach|pragma|load_extension|readfile|writefile|insert|update|delete|drop|create|alter|replace|vacuum)\b/i.test(cleaned)) {
    throw new Error("Only read-only SELECT queries are allowed.");
  }
  return cleaned;
}

export function needsApproval(sql: string): boolean {
  const safe = validateSql(sql);
  return /\bfrom\b/i.test(safe) && !/\bwhere\b/i.test(safe);
}
