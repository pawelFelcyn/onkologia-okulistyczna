import React from "react";
import { Eye, Settings, User } from "lucide-react";

export const Topbar: React.FC = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-medical-200 h-16 flex items-center px-6 lg:px-12 justify-between transition-all duration-300">
      <div className="flex items-center gap-3 group cursor-pointer">
        <div className="p-2 bg-medical-900 text-white rounded-lg group-hover:bg-accent transition-colors duration-300">
          <Eye size={20} strokeWidth={2.5} />
        </div>
        <span className="font-semibold text-lg tracking-tight text-medical-900">
          AEye
        </span>
      </div>

      <div className="flex items-center gap-4">
        <button className="p-2 text-medical-500 hover:text-medical-900 hover:bg-medical-100 rounded-full transition-all">
          <Settings size={20} />
        </button>
        <div className="h-8 w-8 bg-medical-200 rounded-full flex items-center justify-center text-medical-500 hover:bg-medical-300 cursor-pointer transition-colors">
          <User size={18} />
        </div>
      </div>
    </header>
  );
};
