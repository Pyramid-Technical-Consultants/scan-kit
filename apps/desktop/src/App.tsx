import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function App() {
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    invoke<string>("version").then((value) => {
      if (active) {
        setVersion(value);
      }
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="flex min-h-svh items-center justify-center p-6">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Scan Kit</CardTitle>
        </CardHeader>
        <CardContent>{version}</CardContent>
      </Card>
    </div>
  );
}
