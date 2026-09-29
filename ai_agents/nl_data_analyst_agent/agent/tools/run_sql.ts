import { defineTool } from "eve/tools";
import { z } from "zod";
import { inspectDatabaseSchema, needsApproval, openDatabase, validateSql } from "../lib/database";
import { reviewSqlRelevance } from "../lib/jev";
import { boundQuestionFor, currentQuestion } from "../lib/question_context";

export default defineTool({
  description: "Execute one read-only SQLite SELECT on the selected database. Broad queries without a WHERE filter require the engineer's approval before execution.",
  inputSchema: z.object({ databaseId: z.string(), sql: z.string().min(1).max(10000) }),
  approval: ({ toolInput }) => needsApproval(toolInput?.sql ?? "") ? "user-approval" : "not-applicable",
  async execute({ databaseId, sql }, ctx) {
    const safe = validateSql(sql);
    const question = boundQuestionFor(currentQuestion.get(), databaseId, ctx.session.turn.id);
    const db = openDatabase(databaseId);
    try {
      const review = await reviewSqlRelevance({ question, sql: safe, databaseSchema: inspectDatabaseSchema(db) });
      if (review.actionable) {
        return { sql: safe, columns: [], rows: [], review, execution: "blocked" as const };
      }
      const statement = db.prepare(safe);
      const rows = db.prepare(`SELECT * FROM (${safe}) LIMIT 201`).all() as Record<string, unknown>[];
      if (rows.length > 200) throw new Error("Query returned more than 200 rows. Add aggregation or LIMIT.");
      return { sql: safe, columns: rows.length ? Object.keys(rows[0]) : statement.columns().map(column => column.name), rows, review, execution: "completed" as const };
    } finally {
      db.close();
    }
  },
});
