import React from 'react';
import { ScanLine, Loader2 } from 'lucide-react';

interface ActionButtonProps {
    onClick: () => void;
    isLoading?: boolean;
    disabled?: boolean;
    label?: string;
}

export const ActionButton: React.FC<ActionButtonProps> = ({
                                                              onClick,
                                                              isLoading = false,
                                                              disabled = false,
                                                              label = "Mark Tumors"
                                                          }) => {
    return (
        <button
            onClick={onClick}
            disabled={disabled || isLoading}
            className={`
        relative w-full overflow-hidden rounded-xl py-4 px-6 flex items-center justify-center gap-3
        font-semibold text-white shadow-md transition-all duration-300
        ${disabled
                ? 'bg-medical-200 text-medical-400 cursor-not-allowed shadow-none'
                : 'bg-medical-900 hover:bg-accent hover:shadow-lg active:scale-[0.98]'
            }
      `}
        >
            {isLoading ? (
                <Loader2 className="animate-spin" size={20} />
            ) : (
                <ScanLine size={20} />
            )}
            <span>{isLoading ? 'Processing...' : label}</span>
        </button>
    );
};