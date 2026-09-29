import { defineTool } from "eve/tools";
import { z } from "zod";
import { inspectDatabaseSchema, openDatabase } from "../lib/database";

export default defineTool({
  description: "Read table names and columns of the currently selected SQLite database.",
  inputSchema: z.object({ databaseId: z.string() }),
  execute({ databaseId }) {
    const db = openDatabase(databaseId);
    try {
      return inspectDatabaseSchema(db);
    } finally {
      db.close();
    }
  },
});
