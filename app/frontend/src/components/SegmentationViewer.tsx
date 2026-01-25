import React, { useEffect, useRef, useState } from "react";
import { X, Eye, EyeOff, Loader, ChevronRight } from "lucide-react";
import type { Detection } from "../types/inference";
import AEyeLogo from "../assets/logo.svg";

interface SegmentationViewerProps {
  imageUrl: string;
  onClose: () => void;
}

export const SegmentationViewer: React.FC<SegmentationViewerProps> = ({
  imageUrl,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sliderRef = useRef<HTMLDivElement>(null);

  const [showOverlay, setShowOverlay] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [isLoadingYOLO, setIsLoadingYOLO] = useState(true);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [viewMode, setViewMode] = useState<"comparison" | "analysis">(
    "comparison",
  );

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

        const data = await yoloResponse.json();
        setDetections(data.detections || []);
      } catch (error) {
        console.error("API Error:", error);
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
      const maxWidth = window.innerWidth * 0.6;
      const maxHeight = window.innerHeight * 0.75;
      const scale = Math.min(maxWidth / img.width, maxHeight / img.height);

      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      if (viewMode === "comparison") {
        if (showOverlay && detections.length > 0 && sliderPosition > 0) {
          ctx.save();
          const sliderX = (sliderPosition / 100) * canvas.width;
          ctx.beginPath();
          ctx.rect(sliderX, 0, canvas.width - sliderX, canvas.height);
          ctx.clip();
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          drawSegmentation(ctx, canvas.width, canvas.height, scale);
          ctx.restore();
        }
      } else {
        if (showOverlay && detections.length > 0) {
          drawSegmentation(ctx, canvas.width, canvas.height, scale);
        }
      }
      setImageLoaded(true);
    };

    const handleMove = (e: MouseEvent) => {
      if (!sliderRef.current || e.buttons === 0) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      setSliderPosition(Math.max(0, Math.min(100, (x / rect.width) * 100)));
    };

    const handleTouch = (e: TouchEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.touches[0].clientX - rect.left;
      setSliderPosition(Math.max(0, Math.min(100, (x / rect.width) * 100)));
    };

    document.addEventListener("mousemove", handleMove);
    document.addEventListener("touchmove", handleTouch);
    return () => {
      document.removeEventListener("mousemove", handleMove);
      document.removeEventListener("touchmove", handleTouch);
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
    <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-md flex overflow-hidden font-sans text-white">
      <div className="flex-1 flex flex-col relative">
        <header className="p-6 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="flex shrink-0 items-center justify-center p-1.5 bg-white/5 rounded-xl border border-white/10 shadow-sm backdrop-blur-sm">
              <img src={AEyeLogo} alt="AEye logo" className="h-9 w-auto" />
            </div>
            <div>
              <h2 className="text-xl font-bold tracking-tight">
                AI Segmentation
              </h2>
              <p className="text-xs text-white/40 uppercase tracking-widest font-mono">
                Yolov8
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 bg-white/5 border border-white/10 p-1.5 rounded-2xl">
            <div className="flex bg-black/40 p-1 rounded-xl">
              <button
                onClick={() => setViewMode("comparison")}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${viewMode === "comparison" ? "bg-white text-black shadow-lg" : "text-white/50 hover:text-white"}`}
              >
                Compare
              </button>
              <button
                onClick={() => setViewMode("analysis")}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${viewMode === "analysis" ? "bg-white text-black shadow-lg" : "text-white/50 hover:text-white"}`}
              >
                Analysis
              </button>
            </div>

            {viewMode === "analysis" && (
              <div className="ml-4">
                <button
                  onClick={() => setShowOverlay(!showOverlay)}
                  className={`flex items-center gap-2 px-4 py-1.5
                     rounded-xl text-xs font-bold transition-all border ${
                       showOverlay
                         ? `
                            text-white
                            border-accent/50
                            bg-gradient-to-r
                            from-accent-dark
                            via-accent
                            to-accent-light
                            shadow-[0_0_14px_rgba(56,189,248,0.45)]
                          `
                         : `
                            border-white/10
                            text-white/50
                            hover:text-white
                            hover:border-white/30
                          `
                     }`}
                >
                  {showOverlay ? <Eye size={16} /> : <EyeOff size={16} />}
                  {showOverlay ? "Mask ON" : "Mask OFF"}
                </button>
              </div>
            )}
          </div>
        </header>
        <main className="flex-1 flex items-center justify-center p-8 relative">
          <div className="relative rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.8)] border border-white/10 group bg-black">
            {!imageLoaded && (
              <div className="absolute inset-0 flex items-center justify-center z-50 bg-black/50">
                <Loader className="animate-spin text-accent" size={40} />
              </div>
            )}

            <canvas ref={canvasRef} className="block cursor-crosshair" />

            {viewMode === "comparison" &&
              !isLoadingYOLO &&
              detections.length > 0 && (
                <div
                  ref={sliderRef}
                  className="absolute inset-0 cursor-col-resize"
                >
                  <div
                    className="
                              absolute top-0 bottom-0 w-[2px]
                              bg-white
                              transition-[background,box-shadow]


                              group-hover:bg-gradient-to-b
                              group-hover:from-accent-dark
                              group-hover:via-accent
                              group-hover:to-accent-light

                              shadow-[0_0_15px_rgba(255,255,255,0.4)]
                              group-hover:shadow-[0_0_18px_rgba(56,189,248,0.6)]
                            "
                    style={{ left: `${sliderPosition}%` }}
                  >
                    <div
                      className="
                            absolute top-1/2 left-1/2
                            -translate-x-1/2 -translate-y-1/2

                            bg-white text-black
                            rounded-full p-2
                            shadow-2xl border border-black/20

                            transition-transform
                            group-hover:scale-110
                            group-hover:bg-gradient-to-br
                            group-hover:from-accent-dark
                            group-hover:via-accent
                            group-hover:to-accent-light
                            group-hover:text-white
                          "
                    >
                      <ChevronRight size={18} className="rotate-0" />
                    </div>
                  </div>
                  <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-tighter border border-white/10">
                    Original
                  </div>
                  <div
                    className="
                              absolute top-4 right-4
                              px-3 py-1 rounded-full
                              text-[10px] font-bold uppercase tracking-tight

                              bg-black/30 backdrop-blur-md
                              border border-white/10

                              bg-gradient-to-r from-accent-dark via-accent to-accent-light
                              bg-clip-text text-transparent
  "
                  >
                    Segmented
                  </div>
                </div>
              )}
          </div>
        </main>
      </div>
      <aside className="w-[350px] border-l border-white/10 bg-white/[0.07] backdrop-blur-xl flex flex-col">
        <div className="p-8 border-b border-white/10">
          <h3 className="text-sm font-bold uppercase tracking-[0.2em] text-white/40 mb-1">
            Inference Results
          </h3>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black">{detections.length}</span>
            <span className="text-sm text-white/60">Objects detected</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
          {isLoadingYOLO ? (
            <div className="flex flex-col items-center justify-center h-40 opacity-40">
              <Loader className="animate-spin mb-4" />
              <p className="text-sm font-medium">Processing scan...</p>
            </div>
          ) : detections.length > 0 ? (
            detections.map((det, i) => (
              <div
                key={i}
                className="bg-white/5 border border-white/5 rounded-2xl p-4 hover:bg-white/10 transition-all group/card hover:border-accent/30"
              >
                <div className="flex justify-between items-start mb-3">
                  <span
                    className="
                              text-[10px] font-bold uppercase tracking-widest
                              px-2 py-1 rounded-md
                              text-white
                              bg-gradient-to-r
                              from-accent-dark
                              via-accent
                              to-accent-light
                              shadow-[0_0_10px_rgba(56,189,248,0.35)]
                          "
                  >
                    {det.class}
                  </span>
                  <span className="font-mono text-xs text-white/40 group-hover/card:text-white transition-colors">
                    {(det.conf * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="
                                  h-full
                                  bg-gradient-to-r
                                  from-accent-dark
                                  via-accent
                                  to-accent-light
                                  shadow-[0_0_12px_rgba(56,189,248,0.45)]
                                  transition-[width] duration-1000 ease-out
                                "
                      style={{ width: `${det.conf * 100}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-white/30 font-mono">
                    REGION: [{det.box.map((b) => Math.round(b)).join(", ")}]
                  </p>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-20 opacity-30 text-sm italic">
              No clinical anomalies found
            </div>
          )}
        </div>

        <div className="p-6">
          <button
            onClick={onClose}
            className="w-full bg-white text-black py-4 rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-accent transition-colors"
          >
            <X size={18} />
            Close Viewer
          </button>
        </div>
      </aside>
    </div>
  );
};
