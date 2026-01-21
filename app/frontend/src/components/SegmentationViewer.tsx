import React, { useEffect, useRef, useState } from 'react';
import { X, Eye, EyeOff, Layers } from 'lucide-react';
import type { Detection } from '../types/inference';

interface SegmentationViewerProps {
    imageUrl: string;
    detections: Detection[];
    onClose: () => void;
    // In a real app, you would pass the specific AI result data here
    // e.g., predictions: { x: number, y: number, w: number, h: number }[]
}

export const SegmentationViewer: React.FC<SegmentationViewerProps> = ({ imageUrl, detections, onClose }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [showOverlay, setShowOverlay] = useState(true);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            // 1. Setup Canvas Dimensions
            // We scale the canvas to fit the window but keep aspect ratio
            const maxWidth = window.innerWidth * 0.8;
            const maxHeight = window.innerHeight * 0.8;
            const scale = Math.min(maxWidth / img.width, maxHeight / img.height);

            canvas.width = img.width * scale;
            canvas.height = img.height * scale;

            // 2. Draw Original Image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // 3. Draw AI Segmentation (Simulated)
            if (showOverlay && detections.length > 0) {
                drawSegmentation(ctx, canvas.width, canvas.height, scale);
            }

            setImageLoaded(true);
        };
    }, [imageUrl, showOverlay]);

    // --- SIMULATION OF AI OUTPUT ---
    const drawSegmentation = (ctx: CanvasRenderingContext2D, canvasWidth: number, canvasHeight: number, scale: number) => {
        detections.forEach((detection) => {
            if (detection.segments && detection.segments.length > 0) {
                
                ctx.fillStyle = 'rgba(239, 68, 68, 0.4)'; 
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2;

                ctx.beginPath();

                const firstPoint = detection.segments[0];
                ctx.moveTo(
                    firstPoint[0] * canvasWidth,
                    firstPoint[1] * canvasHeight
                );

                for (let i = 0; i< detection.segments.length; i++) {
                    const point = detection.segments[i];
                    ctx.lineTo(
                        point[0] * canvasWidth,
                        point[1] * canvasHeight
                    );
                }

                ctx.closePath();
                ctx.fill();
                ctx.stroke();
            }

            const [x1, y1, x2, y2] = detection.box;
            const boxX = x1 * scale;
            const boxY = y1 * scale;
            const boxWidth = (x2 - x1) * scale;
            const boxHeight = (y2 - y1) * scale;

            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 2;
            ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 14px Inter, sans-serif';
            ctx.fillText(
                `${detection.class} (${(detection.conf * 100).toFixed(1)}%)`,
                boxX + 10,
                boxY - 10
            );
        });

        // Simulating a tumor shape (In reality, this comes from your backend mask coordinates)
        // We draw an irregular blob in the center
        // const cx = w / 2;
        // const cy = h / 2;

        // ctx.moveTo(cx - 50, cy - 20);
        // ctx.bezierCurveTo(cx - 50, cy - 80, cx + 60, cy - 80, cx + 60, cy - 20);
        // ctx.bezierCurveTo(cx + 100, cy, cx + 60, cy + 80, cx, cy + 80);
        // ctx.bezierCurveTo(cx - 60, cy + 80, cx - 80, cy, cx - 50, cy - 20);

        // ctx.fill(); // For U-Net (Segmentation)
        // ctx.stroke(); // For boundary definition

        // // STYLE: YOLO (Bounding Box) style - optional
        // // ctx.strokeRect(cx - 60, cy - 80, 140, 170);

        // // Label
        // ctx.fillStyle = '#ef4444';
        // ctx.font = 'bold 14px Inter, sans-serif';
        // ctx.fillText("Tumor (0.98)", cx + 70, cy - 60);
    };

    return (
        <div className="fixed inset-0 z-[100] bg-medical-900/95 backdrop-blur-sm flex items-center justify-center animate-fade-in">
            {/* Toolbar */}
            <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-center text-white">
                <div className="flex items-center gap-3">
                    <Layers className="text-accent" />
                    <div>
                        <h2 className="text-lg font-semibold">Segmentation Viewer</h2>
                        <p className="text-xs text-medical-200 opacity-70">AI Model: Retina-U-Net-V2</p>
                        <p className="text-xs text-medical-200 opacity-70">
                            Detections: {detections.length}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-4 bg-white/10 rounded-full px-4 py-2 backdrop-blur-md border border-white/10">
                    <button
                        onClick={() => setShowOverlay(!showOverlay)}
                        className="flex items-center gap-2 text-sm font-medium hover:text-accent transition-colors"
                    >
                        {showOverlay ? <Eye size={18} /> : <EyeOff size={18} />}
                        {showOverlay ? "Hide Mask" : "Show Mask"}
                    </button>
                </div>

                <button
                    onClick={onClose}
                    className="p-2 bg-white/10 hover:bg-red-500/80 rounded-full transition-colors"
                >
                    <X size={24} />
                </button>
            </div>

            <div className="relative rounded-lg overflow-hidden shadow-2xl border border-white/10">
                {!imageLoaded && (
                    <div className="absolute inset-0 flex items-center justify-center text-white">
                        Loading...
                    </div>
                )}
                <canvas ref={canvasRef} className="block bg-black" />
            </div>

            <div className="absolute bottom-8 left-8 bg-white/10 backdrop-blur-md p-5 rounded-xl border border-white/10 w-64 shadow-2xl">
                <h4 className="text-white text-center text-xs font-bold uppercase tracking-widest mb-4 border-b border-white/10 pb-2 opacity-80">
                    Analysis Results
                </h4>

             {detections.length > 0 ? (
                    <div className="space-y-4">
                        {detections.map((det, idx) => (
                            <div key={idx} className="bg-white/5 p-3 rounded-lg border border-white/10">
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
                                    <span className="text-medical-200">Polygons:</span>
                                    <span className="font-mono text-white text-right">
                                        {det.segments.length} pts
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-medical-200 text-sm text-center">No detections</p>
                )}
            </div>
        </div>
    );
};