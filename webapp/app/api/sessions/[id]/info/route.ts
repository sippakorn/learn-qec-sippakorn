import { NextResponse } from "next/server";
import { getSessionInfo } from "@/lib/replayer";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const info = await getSessionInfo(id);
    return NextResponse.json(info);
  } catch (err) {
    console.error(`GET /api/sessions/${id}/info failed:`, err);
    return NextResponse.json({ error: "Session not found" }, { status: 404 });
  }
}
