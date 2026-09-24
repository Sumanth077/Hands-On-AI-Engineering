import { defineTool } from "eve/tools";
import { z } from "zod";
import { needsApproval, openDatabase, validateSql } from "../lib/database";

export default defineTool({
  description: "Execute one read-only SQLite SELECT on the selected database. Broad queries without a WHERE filter require the engineer's approval before execution.",
  inputSchema: z.object({ databaseId: z.string(), sql: z.string().min(1).max(10000) }),
  approval: ({ toolInput }) => needsApproval(toolInput?.sql ?? "") ? "user-approval" : "not-applicable",
  execute({ databaseId, sql }) {
    const safe = validateSql(sql);
    const db = openDatabase(databaseId);
    try {
      const statement = db.prepare(safe);
      const rows = db.prepare(`SELECT * FROM (${safe}) LIMIT 201`).all() as Record<string, unknown>[];
      if (rows.length > 200) throw new Error("Query returned more than 200 rows. Add aggregation or LIMIT.");
      return { sql: safe, columns: rows.length ? Object.keys(rows[0]) : statement.columns().map(column => column.name), rows };
    } finally {
      db.close();
    }
  },
});
