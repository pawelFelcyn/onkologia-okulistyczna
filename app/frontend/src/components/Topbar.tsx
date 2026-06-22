import React from "react";
import { Loader2, LogOut, MoonStar, Settings, SunMedium, User } from "lucide-react";
import AEyeLogo from "../assets/logo.svg";

export interface TopbarProps {
  userName: string;
  theme: "light" | "dark";
  isSettingsOpen: boolean;
  isLoggingOut: boolean;
  onOpenSettings: () => void;
  onCloseSettings: () => void;
  onThemeChange: (theme: "light" | "dark") => void;
  onLogout: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({
  userName,
  theme,
  isSettingsOpen,
  isLoggingOut,
  onOpenSettings,
  onCloseSettings,
  onThemeChange,
  onLogout,
}) => {
  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50 flex h-16 items-center justify-between border-b border-medical-200 bg-black/90 px-6 backdrop-blur-md transition-all duration-300 lg:px-12">
        <div className="flex items-center gap-3 group cursor-pointer">
          <img src={AEyeLogo} alt="AEye logo" className="h-9 w-auto" />
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={isSettingsOpen ? onCloseSettings : onOpenSettings}
            className="rounded-full p-2 text-medical-100 transition-all hover:bg-medical-100 hover:text-medical-900"
            aria-haspopup="dialog"
            aria-expanded={isSettingsOpen}
            aria-label="Open appearance settings"
          >
            <Settings size={20} />
          </button>

          <button
            type="button"
            onClick={onLogout}
            disabled={isLoggingOut}
            className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm text-medical-100 transition-colors hover:border-white/20 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Log out"
          >
            {isLoggingOut ? <Loader2 size={16} className="animate-spin" /> : <User size={16} />}
            <span className="hidden sm:inline">{userName}</span>
            <LogOut size={14} />
          </button>
        </div>
      </header>

      {isSettingsOpen && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-[55] bg-black/20"
            onClick={onCloseSettings}
            aria-label="Close appearance settings"
          />

          <div className="fixed right-6 top-20 z-[60] w-[320px] rounded-3xl border border-medical-200 bg-white p-5 shadow-2xl lg:right-12">
            <div>
              <h3 className="text-base font-semibold text-medical-900">Appearance</h3>
              <p className="mt-1 text-sm text-medical-500">
                Choose between light mode and dark mode.
              </p>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => onThemeChange("light")}
                className={`rounded-2xl border px-4 py-4 text-left transition ${
                  theme === "light"
                    ? "border-accent bg-accent/10 text-medical-900"
                    : "border-medical-200 bg-medical-50 text-medical-600 hover:border-medical-300 hover:bg-white"
                }`}
              >
                <SunMedium size={18} />
                <div className="mt-3 text-sm font-semibold">Light</div>
                <div className="mt-1 text-xs text-medical-500">Bright clinical workspace</div>
              </button>

              <button
                type="button"
                onClick={() => onThemeChange("dark")}
                className={`rounded-2xl border px-4 py-4 text-left transition ${
                  theme === "dark"
                    ? "border-accent bg-accent/10 text-medical-900"
                    : "border-medical-200 bg-medical-50 text-medical-600 hover:border-medical-300 hover:bg-white"
                }`}
              >
                <MoonStar size={18} />
                <div className="mt-3 text-sm font-semibold">Dark</div>
                <div className="mt-1 text-xs text-medical-500">Low-glare review mode</div>
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
};
