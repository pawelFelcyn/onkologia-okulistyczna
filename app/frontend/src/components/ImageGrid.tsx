import React from 'react';
import { X } from 'lucide-react';

interface ImageGridProps {
    images: string[];
    onRemove: (index: number) => void;
}

export const ImageGrid: React.FC<ImageGridProps> = ({ images, onRemove }) => {
    if (images.length === 0) return null;

    return (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 animate-fade-in">
            {images.map((src, index) => (
                <div
                    key={index}
                    className="group relative aspect-square bg-black rounded-xl overflow-hidden shadow-sm border border-medical-200 hover:shadow-md transition-all duration-300"
                >
                    <img
                        src={src}
                        alt={`Scan ${index + 1}`}
                        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                    />

                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-3">
                        <span className="text-white text-xs font-medium">Scan {index + 1}</span>
                    </div>

                    <button
                        onClick={() => onRemove(index)}
                        className="absolute top-2 right-2 p-1.5 bg-white/20 backdrop-blur-sm hover:bg-red-500/80 rounded-full text-white opacity-0 group-hover:opacity-100 transition-all duration-200 transform translate-y-[-10px] group-hover:translate-y-0"
                        aria-label="Remove image"
                    >
                        <X size={14} />
                    </button>
                </div>
            ))}
        </div>
    );
};