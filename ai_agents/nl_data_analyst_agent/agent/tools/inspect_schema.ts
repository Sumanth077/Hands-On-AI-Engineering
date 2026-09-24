import { defineTool } from "eve/tools";
import { z } from "zod";
import { openDatabase } from "../lib/database";

export default defineTool({
  description: "Read table names and columns of the currently selected SQLite database.",
  inputSchema: z.object({ databaseId: z.string() }),
  execute({ databaseId }) {
    const db = openDatabase(databaseId);
    try {
      const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").all() as { name: string }[];
      return tables.map(({ name }) => {
        const quoted = `"${name.replaceAll('"', '""')}"`;
        const columns = db.prepare(`PRAGMA table_info(${quoted})`).all() as { name: string; type: string }[];
        return { table: name, columns: columns.map(({ name, type }) => ({ name, type })) };
      });
    } finally {
      db.close();
    }
  },
});
