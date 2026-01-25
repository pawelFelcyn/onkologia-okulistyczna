import React from "react";
import { Eye, Settings, User } from "lucide-react";
import AEyeLogo from "../assets/AEye.svg";

export const Topbar: React.FC = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-md border-b border-medical-200 h-16 flex items-center px-6 lg:px-12 justify-between transition-all duration-300">
      <div className="flex items-center gap-3 group cursor-pointer">
        <img src={AEyeLogo} alt="AEye logo" className="h-9 w-auto" />
      </div>

      <div className="flex items-center gap-4">
        <button className="p-2 text-medical-100 hover:text-medical-900 hover:bg-medical-100 rounded-full transition-all">
          <Settings size={20} />
        </button>
        <div className="h-8 w-8 text-medical-100 hover:text-medical-900 hover:bg-medical-100 rounded-full flex items-center justify-center cursor-pointer transition-colors">
          <User size={18} />
        </div>
      </div>
    </header>
  );
};
