import React, { type ReactNode } from "react";
import { Topbar, type TopbarProps } from "./Topbar";

interface LayoutProps {
  children: ReactNode;
  topbarProps: TopbarProps;
}

export const Layout: React.FC<LayoutProps> = ({ children, topbarProps }) => {
  return (
    <div className="min-h-screen bg-medical-50 flex flex-col">
      <Topbar {...topbarProps} />
      <main className="flex-1 w-full max-w-[1600px] mx-auto pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
};
