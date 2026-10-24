import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PageLoad } from "@/page-load";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <PageLoad>
      <App />
    </PageLoad>
  </React.StrictMode>,
);
