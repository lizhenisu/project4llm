import { WorkspacePage } from "./app/WorkspacePage";
import { AuthProvider } from "./lib/AuthContext";

export function App() {
  return (
    <AuthProvider>
      <WorkspacePage />
    </AuthProvider>
  );
}
