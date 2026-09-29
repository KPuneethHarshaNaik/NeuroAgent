import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Landing } from "@/landing/Landing";
import "@/index.css";

const container = document.getElementById("root");
if (!container) throw new Error("Landing page root element is missing.");

createRoot(container).render(
  <StrictMode>
    <Landing />
  </StrictMode>,
);
