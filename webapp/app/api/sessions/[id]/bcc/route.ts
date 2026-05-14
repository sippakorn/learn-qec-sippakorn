import { downloadBlobOptional } from "@/lib/azure-blob";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const blobName = `bcc_states/h_active_${id}.npz`;

  const buf = await downloadBlobOptional(blobName);
  if (!buf) {
    return new Response(JSON.stringify({ error: "BCC state not found for this session" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(buf.buffer as ArrayBuffer, {
    headers: {
      "Content-Type": "application/zip",
      "Content-Disposition": `attachment; filename="h_active_${id}.npz"`,
      "Content-Length": String(buf.byteLength),
    },
  });
}
