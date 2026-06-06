import React from "react";
import { LineChart } from "lucide-react";
import type { VolumeTrendPoint } from "../types/app";

interface VolumeTrendChartProps {
  points: VolumeTrendPoint[];
}

const WIDTH = 640;
const HEIGHT = 260;
const PADDING_X = 56;
const PADDING_Y = 36;

function formatAxisDate(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
  }).format(new Date(value));
}

function formatVolume(value: number) {
  return value.toFixed(2);
}

export const VolumeTrendChart: React.FC<VolumeTrendChartProps> = ({ points }) => {
  const sortedPoints = [...points].sort((left, right) =>
    left.study_date.localeCompare(right.study_date),
  );

  if (sortedPoints.length === 0) {
    return (
      <div className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-2xl bg-medical-100 p-3 text-medical-800">
            <LineChart size={20} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-medical-900">
              Tumor volume trend
            </h3>
            <p className="text-sm text-medical-500">
              The chart appears once the patient has at least one saved study.
            </p>
          </div>
        </div>
        <div className="flex h-48 items-center justify-center rounded-2xl border border-dashed border-medical-200 bg-medical-50 text-sm text-medical-500">
          No historical volume data available yet.
        </div>
      </div>
    );
  }

  const volumes = sortedPoints.map((point) => point.volume_mm3);
  const minVolume = Math.min(...volumes);
  const maxVolume = Math.max(...volumes);
  const range = maxVolume - minVolume || 1;
  const usableWidth = WIDTH - PADDING_X * 2;
  const usableHeight = HEIGHT - PADDING_Y * 2;

  const graphPoints = sortedPoints.map((point, index) => {
    const x =
      sortedPoints.length === 1
        ? WIDTH / 2
        : PADDING_X + (usableWidth * index) / (sortedPoints.length - 1);
    const y =
      HEIGHT -
      PADDING_Y -
      ((point.volume_mm3 - minVolume) / range) * usableHeight;
    return { ...point, x, y };
  });

  const path = graphPoints
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  const topAxis = maxVolume + range * 0.1;
  const bottomAxis = Math.max(0, minVolume - range * 0.1);

  return (
    <div className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="rounded-2xl bg-medical-100 p-3 text-medical-800">
            <LineChart size={20} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-medical-900">
              Tumor volume trend
            </h3>
            <p className="text-sm text-medical-500">
              Historical algorithm output stored for this patient.
            </p>
          </div>
        </div>
        <div className="rounded-2xl border border-medical-200 bg-medical-50 px-4 py-2 text-right">
          <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
            Latest
          </div>
          <div className="text-lg font-semibold text-medical-900">
            {formatVolume(sortedPoints[sortedPoints.length - 1].volume_mm3)}
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="min-w-[640px]"
          role="img"
          aria-label="Tumor volume change over time"
        >
          <line
            x1={PADDING_X}
            y1={PADDING_Y}
            x2={PADDING_X}
            y2={HEIGHT - PADDING_Y}
            stroke="#cbd5e1"
            strokeWidth="1"
          />
          <line
            x1={PADDING_X}
            y1={HEIGHT - PADDING_Y}
            x2={WIDTH - PADDING_X}
            y2={HEIGHT - PADDING_Y}
            stroke="#cbd5e1"
            strokeWidth="1"
          />

          <text x={16} y={PADDING_Y + 4} fill="#64748b" fontSize="12">
            {formatVolume(topAxis)}
          </text>
          <text
            x={16}
            y={HEIGHT - PADDING_Y + 4}
            fill="#64748b"
            fontSize="12"
          >
            {formatVolume(bottomAxis)}
          </text>

          <path
            d={path}
            fill="none"
            stroke="#38bdf8"
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {graphPoints.map((point) => (
            <g key={point.study_id}>
              <circle
                cx={point.x}
                cy={point.y}
                r="6"
                fill="#7C3AED"
                stroke="#ffffff"
                strokeWidth="3"
              />
              <text
                x={point.x}
                y={HEIGHT - PADDING_Y + 22}
                fill="#475569"
                fontSize="12"
                textAnchor="middle"
              >
                {formatAxisDate(point.study_date)}
              </text>
              <text
                x={point.x}
                y={point.y - 14}
                fill="#0f172a"
                fontSize="12"
                fontWeight="600"
                textAnchor="middle"
              >
                {point.volume_mm3.toFixed(1)}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
};