import React from "react";
import { X, ZoomIn, Check } from "lucide-react";

interface ImageGridProps {
  images: string[];
  selectedIndices?: Set<number>;
  onToggleSelect?: (index: number) => void;
  onRemove: (index: number) => void;
  onImageClick?: (src: string) => void;
}

export const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  selectedIndices = new Set(),
  onToggleSelect,
  onRemove,
  onImageClick,
}) => {
  if (images.length === 0) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 animate-fade-in">
      {images.map((src, index) => {
        const isSelected = selectedIndices.has(index);

        return (
          <div
            key={index}
            onClick={() => onToggleSelect?.(index)}
            className={`
                            group relative aspect-square rounded-xl overflow-hidden
                            border-2 transition-all duration-300 cursor-pointer
                            ${
                              isSelected
                                ? "border-[#7C3AED] ring-1 ring-[#38BDF8] shadow-lg"
                                : // "border-purple-400 ring-2 ring-black/60 shadow-lg"
                                  // "border-purple-400 ring-2 ring-black/60 shadow-lg"

                                  // "border-[#38BDF8] ring-2 ring-[#22D3EE] shadow-lg"
                                  "border-medical-200 hover:ring-2 hover:ring-accent/40"
                            }
                        `}
          >
            {/* Image */}
            <img
              src={src}
              alt={`Scan ${index + 1}`}
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            />

            {/* Hover overlay */}
            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity" />

            {/* View segmentation CTA */}
            {onImageClick && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onImageClick(src);
                }}
                className="
                                    absolute bottom-2 right-2
                                    flex items-center gap-1
                                    px-3 py-1.5
                                    bg-black/70 hover:bg-black/90
                                    text-white text-xs font-medium
                                    rounded-lg opacity-60 group-hover:opacity-100
                                    transition-all
                                "
              >
                <ZoomIn size={14} />
                Analyze
              </button>
            )}

            {/* Selected indicator */}
            {isSelected && (
              <div className="absolute top-2 left-2 flex items-center gap-1 px-2 py-1 rounded-full bg-black text-white text-xs font-semibold">
                <Check size={14} />
                Selected
              </div>
            )}

            {/* Remove button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove(index);
              }}
              className="
                                absolute top-2 right-2 p-1.5
                                bg-white/20 backdrop-blur-sm
                                hover:bg-red-500/80
                                rounded-full text-white
                                opacity-0 group-hover:opacity-100
                                transition-all
                            "
              title="Remove image"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
};
