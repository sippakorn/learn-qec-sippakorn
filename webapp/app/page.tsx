import { listSessions } from "@/lib/azure-table";
import ReplayViewer from "@/components/ReplayViewer";

export const dynamic = "force-dynamic";

export default async function Home() {
  const sessions = await listSessions();

  if (sessions.length === 0) {
    return (
      <main className="min-h-screen bg-[#0f0e17] text-gray-400 flex items-center justify-center font-mono">
        <div className="text-center space-y-2">
          <p className="text-lg">No sessions found in Azure Table Storage.</p>
          <p className="text-sm text-gray-600">Run the upload script first.</p>
        </div>
      </main>
    );
  }

  return <ReplayViewer sessions={sessions} />;
}
