import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Tool } from "@/tool/Tool";
import "@/index.css";

const container = document.getElementById("root");
if (!container) throw new Error("Tool page root element is missing.");

createRoot(container).render(
  <StrictMode>
    <Tool />
  </StrictMode>,
);
