import React, { useRef, useState } from "react";
import { UploadCloud, FileImage } from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface FileUploaderProps {
  onFilesSelected: (files: FileList | null) => void;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onFilesSelected,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
  };

  return (
    <div
      className={twMerge(
        clsx(
          "relative group cursor-pointer flex flex-col items-center justify-center w-full h-64 rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out bg-white",
          isDragging
            ? "border-accent bg-accent/5 scale-[1.01] shadow-lg"
            : "border-medical-200 hover:border-medical-300 hover:bg-medical-50",
        ),
      )}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="image/*"
        className="hidden"
        onChange={handleChange}
      />

      <div className="bg-medical-100 p-4 rounded-full mb-4 group-hover:bg-accent/10 transition-colors duration-300">
        <UploadCloud
          className={clsx(
            "text-medical-400 group-hover:text-accent transition-colors duration-300",
            isDragging && "text-accent",
          )}
          size={32}
        />
      </div>

      <h3 className="text-lg font-medium text-medical-800 mb-1">
        Upload OCT Scans
      </h3>
      <p className="text-sm text-medical-500">
        Drag & drop or click to select files
      </p>

      <div className="absolute bottom-4 flex items-center gap-2 text-xs text-medical-400 opacity-60">
        <FileImage size={14} />
        <span>Supports JPG, JPEG, PNG,</span>
      </div>
    </div>
  );
};
