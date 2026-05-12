import { BlobServiceClient } from "@azure/storage-blob";

function containerClient() {
  return BlobServiceClient.fromConnectionString(
    process.env.AZURE_STORAGE_CONNECTION_STRING!
  ).getContainerClient(process.env.AZURE_BLOB_CONTAINER ?? "replay-data");
}

export async function downloadBlob(blobName: string): Promise<Buffer> {
  return containerClient().getBlobClient(blobName).downloadToBuffer();
}

export async function listBlobs(prefix: string): Promise<string[]> {
  const names: string[] = [];
  for await (const item of containerClient().listBlobsFlat({ prefix })) {
    names.push(item.name);
  }
  return names;
}
