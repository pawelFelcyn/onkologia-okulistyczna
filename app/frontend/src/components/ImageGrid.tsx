import React from 'react';
import { X, ZoomIn } from 'lucide-react';

interface ImageGridProps {
    images: string[];
    onRemove: (index: number) => void;
    onImageClick?: (src: string) => void;
}

export const ImageGrid: React.FC<ImageGridProps> = ({ images, onRemove, onImageClick }) => {
    if (images.length === 0) return null;

    return (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 animate-fade-in">
            {images.map((src, index) => (
                <div
                    key={index}
                    onClick={() => onImageClick && onImageClick(src)}
                    className={`
            group relative aspect-square bg-black rounded-xl overflow-hidden shadow-sm border border-medical-200 
            transition-all duration-300
            ${onImageClick ? 'cursor-pointer hover:shadow-lg hover:ring-2 hover:ring-accent' : ''}
          `}
                >
                    <img
                        src={src}
                        alt={`Scan ${index + 1}`}
                        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                    />

                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                        <ZoomIn className="text-white" size={24} />
                    </div>

                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onRemove(index);
                        }}
                        className="absolute top-2 right-2 p-1.5 bg-white/20 backdrop-blur-sm hover:bg-red-500/80 rounded-full text-white opacity-0 group-hover:opacity-100 transition-all z-10"
                    >
                        <X size={14} />
                    </button>
                </div>
            ))}
        </div>
    );
};