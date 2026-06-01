import type { Detection } from "./inference";

export interface Patient {
  id: number;
  first_name: string;
  last_name: string;
  full_name: string;
  created_at: string;
  study_count: number;
  latest_study_date: string | null;
}

export interface StudySummary {
  id: number;
  patient_id: number;
  study_date: string;
  selected_model: string;
  volume_mm3: number;
  created_at: string;
  scan_count: number;
}

export interface VolumeTrendPoint {
  study_id: number;
  study_date: string;
  volume_mm3: number;
}

export interface StudyScan {
  id: number;
  filename: string;
  image_url: string;
  detections: Detection[];
  error?: string | null;
  sort_order: number;
  created_at: string;
}

export interface StudyDetail extends StudySummary {
  patient_name?: string;
  scans: StudyScan[];
}