import React, { useEffect, useRef, useState } from "react";
import { X, Eye, EyeOff, Layers, Loader } from "lucide-react";
import type { Detection } from "../types/inference";

interface SegmentationViewerProps {
  imageUrl: string;
  onClose: () => void;
}

export const SegmentationViewer: React.FC<SegmentationViewerProps> = ({
  imageUrl,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [isLoadingYOLO, setIsLoadingYOLO] = useState(true);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [viewMode, setViewMode] = useState<"comparison" | "analysis">(
    "comparison",
  );
  const sliderRef = useRef<HTMLDivElement>(null);

  // Fetch YOLO segmentation when component mounts
  useEffect(() => {
    const fetchYOLOSegmentation = async () => {
      setIsLoadingYOLO(true);
      try {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const formData = new FormData();
        formData.append("file", blob);

        const yoloResponse = await fetch("http://localhost:8000/inference", {
          method: "POST",
          body: formData,
        });

        if (!yoloResponse.ok) {
          throw new Error("YOLO segmentation failed");
        }

        const data = await yoloResponse.json();
        setDetections(data.detections || []);
      } catch (error) {
        console.error("YOLO API call failed:", error);
        setDetections([]);
      } finally {
        setIsLoadingYOLO(false);
      }
    };

    fetchYOLOSegmentation();
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      // Setup Canvas Dimensions
      const maxWidth = window.innerWidth * 0.7; // 0.8
      const maxHeight = window.innerHeight * 0.75; // 0.8
      const scale = Math.min(maxWidth / img.width, maxHeight / img.height);

      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      // Draw composite based on view mode
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw full image as base
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Determine how to draw based on view mode
      if (viewMode === "comparison") {
        // Comparison Mode: Slider with clip region
        if (showOverlay && detections.length > 0 && sliderPosition > 0) {
          ctx.save();
          const sliderX = (sliderPosition / 100) * canvas.width;
          ctx.beginPath();
          ctx.rect(sliderX, 0, canvas.width - sliderX, canvas.height);
          ctx.clip();
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          drawSegmentation(ctx, canvas.width, canvas.height, scale);
          ctx.restore();
        } else if (!showOverlay && detections.length > 0) {
          drawSegmentation(ctx, canvas.width, canvas.height, scale);
        }
      } else {
        // Analysis Mode: Toggle between full image and image with segmentation
        if (showOverlay && detections.length > 0) {
          drawSegmentation(ctx, canvas.width, canvas.height, scale);
        }
      }

      setImageLoaded(true);
    };

    // Handle mouse move for slider
    const handleMouseMove = (e: MouseEvent) => {
      if (!sliderRef.current || e.buttons === 0) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPosition(percentage);
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!sliderRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.touches[0].clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPosition(percentage);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("touchmove", handleTouchMove);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("touchmove", handleTouchMove);
    };
  }, [imageUrl, showOverlay, sliderPosition, detections, viewMode]);

  const drawSegmentation = (
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
    canvasHeight: number,
    scale: number,
  ) => {
    detections.forEach((detection) => {
      // Draw segmentation mask
      if (detection.segments && detection.segments.length > 0) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.35)";
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2 * scale;

        ctx.beginPath();

        const firstPoint = detection.segments[0];
        ctx.moveTo(firstPoint[0] * canvasWidth, firstPoint[1] * canvasHeight);

        for (let i = 1; i < detection.segments.length; i++) {
          const point = detection.segments[i];
          ctx.lineTo(point[0] * canvasWidth, point[1] * canvasHeight);
          ctx.lineJoin = "round";
          ctx.shadowBlur = 4;
          ctx.shadowColor = "#ef4444";
        }

        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Draw bounding box
      const [x1, y1, x2, y2] = detection.box;
      const boxX = x1 * scale;
      const boxY = y1 * scale;
      const boxWidth = (x2 - x1) * scale;
      const boxHeight = (y2 - y1) * scale;

      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2 * scale;
      ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

      // Draw label
      ctx.fillStyle = "#ef4444";
      ctx.font = `bold ${14 * scale}px Inter, sans-serif`;
      ctx.fillText(
        `${detection.class} (${(detection.conf * 100).toFixed(1)}%)`,
        boxX + 10,
        boxY - 10,
      );
    });
  };

  return (
    <div className="fixed inset-0 z-[100] bg-medical-900/95 backdrop-blur-sm flex items-center justify-center animate-fade-in">
      {/* Toolbar */}
      <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-center text-white">
        <div className="flex items-center gap-3">
          <Layers className="text-accent" />
          <div>
            <h2 className="text-lg font-semibold">Segmentation Viewer</h2>
            <p className="text-xs text-medical-200 opacity-70">
              AI Model: Fine tuned YOLOv8
            </p>
            <p className="text-xs text-medical-200 opacity-70">
              Detections: {detections.length}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-white/10 rounded-full px-4 py-2 backdrop-blur-md border border-white/10">
          {isLoadingYOLO && (
            <div className="flex items-center gap-2 text-sm font-medium text-accent">
              <Loader size={16} className="animate-spin" />
              AI Segmenting...
            </div>
          )}
          {!isLoadingYOLO && (
            <>
              {/* View Mode Switcher */}
              <div className="flex items-center gap-1 bg-white/10 rounded-lg p-1 border border-white/20">
                <button
                  onClick={() => setViewMode("comparison")}
                  className={
                    viewMode === "comparison"
                      ? "px-3 py-1 rounded text-xs font-medium transition-all bg-accent text-medical-900 shadow-md"
                      : "px-3 py-1 rounded text-xs font-medium transition-all text-medical-200 hover:text-white"
                  }
                  title="Compare original and segmented scans side-by-side"
                >
                  Compare
                </button>
                <button
                  onClick={() => setViewMode("analysis")}
                  className={
                    viewMode === "analysis"
                      ? "px-3 py-1 rounded text-xs font-medium transition-all bg-accent text-medical-900 shadow-md"
                      : "px-3 py-1 rounded text-xs font-medium transition-all text-medical-200 hover:text-white"
                  }
                  title="Analyze segmentation with mask toggle"
                >
                  Analyze
                </button>
              </div>

              {/* Mask Toggle - Only in Analysis Mode */}
              {viewMode === "analysis" && (
                <button
                  onClick={() => setShowOverlay(!showOverlay)}
                  className="flex items-center gap-2 text-sm font-medium hover:text-accent transition-colors"
                  title={
                    showOverlay
                      ? "Hide segmentation mask"
                      : "Show segmentation mask"
                  }
                >
                  {showOverlay ? <Eye size={18} /> : <EyeOff size={18} />}
                  {showOverlay ? "Hide Mask" : "Show Mask"}
                </button>
              )}
            </>
          )}
        </div>

        <button
          onClick={onClose}
          className="p-2 bg-white/10 hover:bg-red-500/80 rounded-full transition-colors"
        >
          <X size={24} />
        </button>
      </div>

      {/* Canvas with Slider */}
      <div className="relative rounded-lg overflow-hidden shadow-2xl border border-white/40">
        {!imageLoaded || isLoadingYOLO ? (
          <div className="absolute inset-0 flex items-center justify-center text-white bg-black/50 z-10">
            <div className="text-center">
              <Loader
                size={40}
                className="animate-spin mx-auto mb-3 text-accent"
              />
              <p>AI Segmenting...</p>
            </div>
          </div>
        ) : null}
        <canvas ref={canvasRef} className="block bg-black cursor-col-resize" />
        {/* Before/After Slider - Only in Comparison Mode */}
        {detections.length > 0 &&
          !isLoadingYOLO &&
          viewMode === "comparison" && (
            <div
              ref={sliderRef}
              className="absolute inset-0 cursor-col-resize group"
              onMouseDown={() => {}}
              onTouchStart={() => {}}
            >
              <div
                className="absolute top-0 bottom-0 w-[2px]
                                bg-white/70 hover:bg-white
                                transition-colors"
                style={{ left: `${sliderPosition}%` }}
              >
                {/* <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-accent text-medical-900 rounded-full p-2 shadow-lg"> */}
                <div
                  className="
                            absolute top-1/2 left-1/2
                            -translate-x-1/2 -translate-y-1/2
                            bg-black/20 hover:bg-black/90
                            rounded-full p-2
                            shadow-lg
                            ring-2 ring-white/70
                            "
                >
                  <svg
                    className="w-3 h-3 text-medical-900"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M15.75 5.75L12 2l-3.75 3.75M12 22v-5M8.25 18.25L12 22l3.75-3.75" />
                  </svg>
                </div>
              </div>
              {/* Before label */}
              <div className="absolute top-4 left-4 bg-white/20 backdrop-blur-md px-3 py-1 rounded-full text-xs font-semibold text-white">
                Original
              </div>
              {/* After label */}
              <div className="absolute top-4 right-4 bg-accent/30 backdrop-blur-md px-3 py-1 rounded-full text-xs font-semibold text-white">
                Segmented
              </div>
            </div>
          )}
      </div>

      {/* Analysis Results Panel */}
      <div className="absolute bottom-8 left-8 bg-white/10 backdrop-blur-md p-5 rounded-xl border border-white/10 w-64 shadow-2xl max-h-80 overflow-y-auto">
        <h4 className="text-white text-center text-xs font-bold uppercase tracking-widest mb-4 border-b border-white/10 pb-2 opacity-80">
          Analysis Results
        </h4>

        {isLoadingYOLO ? (
          <div className="flex items-center justify-center py-6">
            <Loader size={24} className="animate-spin text-accent" />
          </div>
        ) : detections.length > 0 ? (
          <div className="space-y-4">
            {detections.map((det, idx) => (
              <div
                key={idx}
                className="bg-white/5 p-3 rounded-lg border border-white/10 hover:bg-white/10 transition-colors"
              >
                <div className="font-semibold text-accent capitalize mb-2">
                  {det.class} #{idx + 1}
                </div>
                <div className="grid grid-cols-[100px_1fr] gap-y-1 text-xs">
                  <span className="text-medical-200">Confidence:</span>
                  <span className="font-mono text-white text-right">
                    {(det.conf * 100).toFixed(1)}%
                  </span>
                  <span className="text-medical-200">Bounding Box:</span>
                  <span className="font-mono text-white text-right text-[10px]">
                    [{det.box[0].toFixed(0)}, {det.box[1].toFixed(0)}]
                  </span>
                  {det.segments && (
                    <>
                      <span className="text-medical-200">Polygons:</span>
                      <span className="font-mono text-white text-right">
                        {det.segments.length} pts
                      </span>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-medical-200 text-sm text-center py-6">
            No detections found
          </p>
        )}
      </div>
    </div>
  );
};
