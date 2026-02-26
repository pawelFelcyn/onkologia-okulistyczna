import React from "react";
import { Activity, AlertCircle } from "lucide-react";

interface SummaryPanelProps {
  hasResults: boolean;
  volume?: number;
}

export const SummaryPanel: React.FC<SummaryPanelProps> = ({
  hasResults,
  volume = 0,
}) => {
  return (
    <div className="bg-white rounded-2xl border border-medical-200 p-6 h-full flex flex-col shadow-sm">
      <div className="flex items-center gap-2 mb-6 border-b border-medical-100 pb-4">
        <Activity className="text-accent" size={20} />
        <h2 className="text-lg font-semibold text-medical-900">
          Algorithm results
        </h2>
      </div>

      {!hasResults ? (
        <div className="flex-1 flex flex-col items-center justify-center text-medical-400 text-center space-y-3 opacity-60">
          <AlertCircle size={40} strokeWidth={1.5} />
          <p className="text-sm">
            Upload images and run analysis to see volumetric data.
          </p>
        </div>
      ) : (
        <div className="flex-1 space-y-6 animate-fade-in">
          <div className="bg-medical-50 rounded-xl p-4 border border-medical-100">
            <span className="text-sm text-medical-500 font-medium uppercase tracking-wide">
              Est. Tumor Volume
            </span>
            <div className="flex items-end gap-2 mt-1">
              <span className="text-4xl font-bold text-medical-900">
                {volume.toFixed(2)}
              </span>
              <span className="text-base text-medical-500 mb-1.5">mm³</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
