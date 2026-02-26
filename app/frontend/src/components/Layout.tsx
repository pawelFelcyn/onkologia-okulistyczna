import React, { type ReactNode } from "react";
import { Topbar } from "./Topbar";

interface LayoutProps {
  children: ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-medical-50 flex flex-col">
      <Topbar />
      <main className="flex-1 w-full max-w-[1600px] mx-auto pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
};
