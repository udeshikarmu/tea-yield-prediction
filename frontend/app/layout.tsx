import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sri Lankan Tea Agro-Climatic Intelligence",
  description: "Explainable AI Harvest Readiness & Yield Estimation Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased selection:bg-emerald-500 selection:text-slate-950">
        {children}
      </body>
    </html>
  );
}
