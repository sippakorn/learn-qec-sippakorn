import { NextResponse } from "next/server";
import { listSessions } from "@/lib/azure-table";

export async function GET() {
  try {
    const sessions = await listSessions();
    return NextResponse.json(sessions);
  } catch (err) {
    console.error("GET /api/sessions failed:", err);
    return NextResponse.json({ error: "Failed to list sessions" }, { status: 500 });
  }
}
