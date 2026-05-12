import { NextResponse } from "next/server";
import { getStep } from "@/lib/replayer";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string; step: string }> }
) {
  const { id, step: stepStr } = await params;
  const step = parseInt(stepStr, 10);

  if (isNaN(step) || step < 0) {
    return NextResponse.json({ error: "Invalid step" }, { status: 400 });
  }

  try {
    const result = await getStep(id, step);
    return NextResponse.json(result);
  } catch (err) {
    console.error(`GET /api/sessions/${id}/step/${step} failed:`, err);
    return NextResponse.json({ error: "Failed to compute step" }, { status: 500 });
  }
}
