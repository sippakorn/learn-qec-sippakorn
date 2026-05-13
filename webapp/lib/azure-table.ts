import { TableClient } from "@azure/data-tables";

function tableClient() {
  return TableClient.fromConnectionString(
    process.env.AZURE_STORAGE_CONNECTION_STRING!,
    process.env.AZURE_TABLE_NAME ?? "sessions"
  );
}

export interface SessionMeta {
  sessionId: string;
  totalSteps: number;
  nCheckpoints: number;
  shapeRows: number;
  shapeCols: number;
  createdAt: string;
}

export async function listSessions(): Promise<SessionMeta[]> {
  const client = tableClient();
  const results: SessionMeta[] = [];

  for await (const entity of client.listEntities({
    queryOptions: { filter: "PartitionKey eq 'session'" },
  })) {
    results.push({
      sessionId:    entity.rowKey as string,
      totalSteps:   entity["total_steps"]   as number,
      nCheckpoints: entity["n_checkpoints"] as number,
      shapeRows:    entity["shape_rows"]    as number,
      shapeCols:    entity["shape_cols"]    as number,
      createdAt:    entity["created_at"]    as string,
    });
  }

  // Newest first by createdAt
  results.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return results;
}

export async function getSession(sessionId: string): Promise<SessionMeta | null> {
  try {
    const entity = await tableClient().getEntity("session", sessionId);
    return {
      sessionId,
      totalSteps:   entity["total_steps"]   as number,
      nCheckpoints: entity["n_checkpoints"] as number,
      shapeRows:    entity["shape_rows"]    as number,
      shapeCols:    entity["shape_cols"]    as number,
      createdAt:    entity["created_at"]    as string,
    };
  } catch {
    return null;
  }
}
