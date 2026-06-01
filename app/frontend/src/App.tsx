import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { CalendarDays, Database, FolderOpen, Loader2, Plus, ScanLine, UserRound } from "lucide-react";
import { Layout } from "./components/Layout";
import { FileUploader } from "./components/FileUploader";
import { SegmentationViewer } from "./components/SegmentationViewer";
import { VolumeTrendChart } from "./components/VolumeTrendChart";
import type { Patient, StudyDetail, StudyScan, StudySummary, VolumeTrendPoint } from "./types/app";

const API_BASE = "http://localhost:8000";

type ModelType = "yolo" | "unet";

interface PendingScan {
  id: string;
  file: File;
  previewUrl: string;
}

function formatDate(value: string | null) {
  if (!value) {
    return "No studies yet";
  }

  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  }).format(new Date(value));
}

function formatDateTime(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatModelName(model: string) {
  return model === "unet" ? "U-Net" : "YOLOv8";
}

function buildApiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function delay(ms: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < 6; attempt += 1) {
    try {
      const response = await fetch(buildApiUrl(path), init);
      const isJson = response.headers
        .get("content-type")
        ?.includes("application/json");
      const payload = isJson ? await response.json() : null;

      if (!response.ok) {
        const detail =
          payload && typeof payload === "object" && "detail" in payload
            ? String(payload.detail)
            : "Request failed";

        if (response.status >= 500 && attempt < 5) {
          await delay(1000 * (attempt + 1));
          continue;
        }

        throw new Error(detail);
      }

      return payload as T;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Request failed");
      if (attempt === 5) {
        break;
      }

      await delay(1000 * (attempt + 1));
    }
  }

  throw lastError ?? new Error("Request failed");
}

function getTumorDetectionCount(scans: StudyScan[]) {
  return scans.reduce((total, scan) => {
    return total + scan.detections.filter((detection) => detection.class === "tumor").length;
  }, 0);
}

function App() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);
  const [selectedStudyId, setSelectedStudyId] = useState<number | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [studies, setStudies] = useState<StudySummary[]>([]);
  const [selectedStudy, setSelectedStudy] = useState<StudyDetail | null>(null);
  const [trendPoints, setTrendPoints] = useState<VolumeTrendPoint[]>([]);
  const [pendingScans, setPendingScans] = useState<PendingScan[]>([]);
  const [viewerScan, setViewerScan] = useState<StudyScan | null>(null);
  const [patientsLoading, setPatientsLoading] = useState(true);
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [studyLoading, setStudyLoading] = useState(false);
  const [creatingPatient, setCreatingPatient] = useState(false);
  const [creatingStudy, setCreatingStudy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [patientForm, setPatientForm] = useState({ firstName: "", lastName: "" });
  const [studyForm, setStudyForm] = useState<{ studyDate: string; model: ModelType }>({
    studyDate: new Date().toISOString().slice(0, 10),
    model: "yolo",
  });

  const loadPatients = async (preferredPatientId?: number) => {
    setPatientsLoading(true);

    try {
      const data = await apiRequest<{ patients: Patient[] }>("/patients");
      setPatients(data.patients);
      setSelectedPatientId((current) => {
        if (preferredPatientId && data.patients.some((patient) => patient.id === preferredPatientId)) {
          return preferredPatientId;
        }
        if (current && data.patients.some((patient) => patient.id === current)) {
          return current;
        }
        return data.patients[0]?.id ?? null;
      });
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to load patients");
    } finally {
      setPatientsLoading(false);
    }
  };

  const loadPatientWorkspace = async (
    patientId: number,
    preferredStudyId?: number,
  ) => {
    setWorkspaceLoading(true);
    setSelectedStudy(null);

    try {
      const [patient, studiesResponse, trendResponse] = await Promise.all([
        apiRequest<Patient>(`/patients/${patientId}`),
        apiRequest<{ studies: StudySummary[] }>(`/patients/${patientId}/studies`),
        apiRequest<{ points?: VolumeTrendPoint[]; trend?: VolumeTrendPoint[] }>(
          `/patients/${patientId}/volume-trend`,
        ),
      ]);

      setSelectedPatient(patient);
      setStudies(studiesResponse.studies);
      setTrendPoints(trendResponse.points ?? trendResponse.trend ?? []);
      setSelectedStudyId((current) => {
        if (preferredStudyId && studiesResponse.studies.some((study) => study.id === preferredStudyId)) {
          return preferredStudyId;
        }
        if (current && studiesResponse.studies.some((study) => study.id === current)) {
          return current;
        }
        return studiesResponse.studies[0]?.id ?? null;
      });
      setErrorMessage(null);
    } catch (error) {
      setSelectedPatient(null);
      setStudies([]);
      setTrendPoints([]);
      setSelectedStudyId(null);
      setErrorMessage(error instanceof Error ? error.message : "Failed to load patient workspace");
    } finally {
      setWorkspaceLoading(false);
    }
  };

  const loadStudy = async (studyId: number) => {
    setStudyLoading(true);
    try {
      const study = await apiRequest<StudyDetail>(`/studies/${studyId}`);
      setSelectedStudy(study);
      setErrorMessage(null);
    } catch (error) {
      setSelectedStudy(null);
      setErrorMessage(error instanceof Error ? error.message : "Failed to load study details");
    } finally {
      setStudyLoading(false);
    }
  };

  useEffect(() => {
    void loadPatients();
  }, []);

  useEffect(() => {
    if (selectedPatientId === null) {
      setSelectedPatient(null);
      setStudies([]);
      setTrendPoints([]);
      setSelectedStudyId(null);
      return;
    }

    void loadPatientWorkspace(selectedPatientId);
  }, [selectedPatientId]);

  useEffect(() => {
    if (selectedStudyId === null) {
      setSelectedStudy(null);
      return;
    }

    void loadStudy(selectedStudyId);
  }, [selectedStudyId]);

  useEffect(() => {
    return () => {
      pendingScans.forEach((scan) => URL.revokeObjectURL(scan.previewUrl));
    };
  }, [pendingScans]);

  const handleFilesSelected = (fileList: FileList | null) => {
    if (!fileList) {
      return;
    }

    const nextScans = Array.from(fileList).map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`,
      file,
      previewUrl: URL.createObjectURL(file),
    }));

    setPendingScans((current) => [...current, ...nextScans]);
  };

  const removePendingScan = (scanId: string) => {
    setPendingScans((current) => {
      const scan = current.find((item) => item.id === scanId);
      if (scan) {
        URL.revokeObjectURL(scan.previewUrl);
      }
      return current.filter((item) => item.id !== scanId);
    });
  };

  const clearPendingScans = () => {
    setPendingScans((current) => {
      current.forEach((scan) => URL.revokeObjectURL(scan.previewUrl));
      return [];
    });
  };

  const handleCreatePatient = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setCreatingPatient(true);

    try {
      const created = await apiRequest<Patient>("/patients", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          first_name: patientForm.firstName,
          last_name: patientForm.lastName,
        }),
      });

      setPatientForm({ firstName: "", lastName: "" });
      await loadPatients(created.id);
      setSelectedPatientId(created.id);
      setSelectedStudyId(null);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to create patient");
    } finally {
      setCreatingPatient(false);
    }
  };

  const handleCreateStudy = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (selectedPatientId === null) {
      setErrorMessage("Select a patient before creating a study");
      return;
    }

    if (pendingScans.length < 3) {
      setErrorMessage("A study requires at least 3 scans");
      return;
    }

    setCreatingStudy(true);

    try {
      const formData = new FormData();
      formData.append("study_date", studyForm.studyDate);
      formData.append("model", studyForm.model);

      pendingScans.forEach((scan) => {
        formData.append("files", scan.file, scan.file.name);
      });

      const created = await apiRequest<StudyDetail>(
        `/patients/${selectedPatientId}/studies`,
        {
          method: "POST",
          body: formData,
        },
      );

      clearPendingScans();
      setSelectedStudy(created);
      await loadPatients(selectedPatientId);
      await loadPatientWorkspace(selectedPatientId, created.id);
      setSelectedStudyId(created.id);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to create study");
    } finally {
      setCreatingStudy(false);
    }
  };

  const selectedStudyTumorCount = selectedStudy
    ? getTumorDetectionCount(selectedStudy.scans)
    : 0;

  return (
    <Layout>
      {viewerScan && selectedStudy && (
        <SegmentationViewer
          imageUrl={buildApiUrl(viewerScan.image_url)}
          detections={viewerScan.detections}
          imageName={viewerScan.filename}
          modelName={formatModelName(selectedStudy.selected_model)}
          studyDate={formatDate(selectedStudy.study_date)}
          errorMessage={viewerScan.error}
          onClose={() => setViewerScan(null)}
        />
      )}

      <div className="grid grid-cols-1 gap-8 xl:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="space-y-6">
          <section className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
            <div className="mb-5 flex items-center gap-3">
              <div className="rounded-2xl bg-medical-100 p-3 text-medical-800">
                <UserRound size={20} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-medical-900">Patients</h2>
                <p className="text-sm text-medical-500">
                  Create and browse patient records.
                </p>
              </div>
            </div>

            <form className="space-y-3" onSubmit={handleCreatePatient}>
              <input
                value={patientForm.firstName}
                onChange={(event) =>
                  setPatientForm((current) => ({
                    ...current,
                    firstName: event.target.value,
                  }))
                }
                className="w-full rounded-2xl border border-medical-200 bg-medical-50 px-4 py-3 text-sm outline-none transition focus:border-accent"
                placeholder="First name"
              />
              <input
                value={patientForm.lastName}
                onChange={(event) =>
                  setPatientForm((current) => ({
                    ...current,
                    lastName: event.target.value,
                  }))
                }
                className="w-full rounded-2xl border border-medical-200 bg-medical-50 px-4 py-3 text-sm outline-none transition focus:border-accent"
                placeholder="Last name"
              />
              <button
                type="submit"
                disabled={creatingPatient}
                className="flex w-full items-center justify-center gap-2 rounded-2xl bg-medical-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-medical-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {creatingPatient ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
                Add patient
              </button>
            </form>
          </section>

          <section className="rounded-3xl border border-medical-200 bg-white p-4 shadow-sm">
            <div className="mb-4 flex items-center justify-between px-2">
              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-medical-500">
                Registry
              </h3>
              <span className="rounded-full bg-medical-100 px-3 py-1 text-xs font-medium text-medical-600">
                {patients.length} patients
              </span>
            </div>

            <div className="max-h-[620px] space-y-3 overflow-y-auto pr-1">
              {patientsLoading ? (
                <div className="flex h-28 items-center justify-center text-sm text-medical-500">
                  <Loader2 size={18} className="mr-2 animate-spin" />
                  Loading patients...
                </div>
              ) : patients.length > 0 ? (
                patients.map((patient) => (
                  <button
                    key={patient.id}
                    onClick={() => setSelectedPatientId(patient.id)}
                    className={`w-full rounded-3xl border p-4 text-left transition ${
                      selectedPatientId === patient.id
                        ? "border-accent bg-accent/10 shadow-sm"
                        : "border-medical-200 bg-medical-50 hover:border-medical-300 hover:bg-white"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-base font-semibold text-medical-900">
                          {patient.full_name}
                        </div>
                        <div className="mt-1 text-sm text-medical-500">
                          Latest study: {formatDate(patient.latest_study_date)}
                        </div>
                      </div>
                      <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-medical-600">
                        {patient.study_count} studies
                      </span>
                    </div>
                  </button>
                ))
              ) : (
                <div className="rounded-3xl border border-dashed border-medical-200 bg-medical-50 px-4 py-8 text-center text-sm text-medical-500">
                  No patients yet. Create the first patient to start saving OCT studies.
                </div>
              )}
            </div>
          </section>
        </aside>

        <section className="space-y-6">
          {errorMessage && (
            <div className="rounded-3xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700 shadow-sm">
              {errorMessage}
            </div>
          )}

          {selectedPatient ? (
            <>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
                <div className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
                  <div className="flex flex-wrap items-start justify-between gap-6">
                    <div>
                      <div className="mb-2 flex items-center gap-3">
                        <div className="rounded-2xl bg-medical-100 p-3 text-medical-800">
                          <FolderOpen size={20} />
                        </div>
                        <div>
                          <h1 className="text-2xl font-semibold text-medical-900">
                            {selectedPatient.full_name}
                          </h1>
                          <p className="text-sm text-medical-500">
                            Persistent patient record with saved studies and tumor volumes.
                          </p>
                        </div>
                      </div>

                      <div className="mt-6 grid gap-3 sm:grid-cols-3">
                        <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                          <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                            Studies
                          </div>
                          <div className="mt-2 text-2xl font-semibold text-medical-900">
                            {selectedPatient.study_count}
                          </div>
                        </div>
                        <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                          <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                            Last saved study
                          </div>
                          <div className="mt-2 text-sm font-semibold text-medical-900">
                            {formatDate(selectedPatient.latest_study_date)}
                          </div>
                        </div>
                        <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                          <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                            Record created
                          </div>
                          <div className="mt-2 text-sm font-semibold text-medical-900">
                            {formatDate(selectedPatient.created_at)}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-3xl border border-medical-200 bg-medical-50 p-5 text-sm text-medical-600">
                      <div className="mb-2 flex items-center gap-2 font-semibold text-medical-900">
                        <Database size={16} />
                        Stored data
                      </div>
                      <p>
                        Each study stores the study date, uploaded scans, segmentation masks and computed tumor volume.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
                  <div className="mb-4 flex items-center gap-3">
                    <div className="rounded-2xl bg-medical-100 p-3 text-medical-800">
                      <Plus size={18} />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-medical-900">
                        New study
                      </h2>
                      <p className="text-sm text-medical-500">
                        Upload a dated sequence for this patient.
                      </p>
                    </div>
                  </div>

                  <form className="space-y-4" onSubmit={handleCreateStudy}>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="space-y-2 text-sm text-medical-700">
                        <span className="font-medium">Study date</span>
                        <input
                          type="date"
                          value={studyForm.studyDate}
                          onChange={(event) =>
                            setStudyForm((current) => ({
                              ...current,
                              studyDate: event.target.value,
                            }))
                          }
                          className="w-full rounded-2xl border border-medical-200 bg-medical-50 px-4 py-3 outline-none transition focus:border-accent"
                        />
                      </label>

                      <label className="space-y-2 text-sm text-medical-700">
                        <span className="font-medium">Segmentation model</span>
                        <select
                          value={studyForm.model}
                          onChange={(event) =>
                            setStudyForm((current) => ({
                              ...current,
                              model: event.target.value as ModelType,
                            }))
                          }
                          className="w-full rounded-2xl border border-medical-200 bg-medical-50 px-4 py-3 outline-none transition focus:border-accent"
                        >
                          <option value="yolo">YOLOv8</option>
                          <option value="unet">U-Net</option>
                        </select>
                      </label>
                    </div>

                    <FileUploader onFilesSelected={handleFilesSelected} />

                    {pendingScans.length > 0 && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium text-medical-900">
                            Pending scans
                          </div>
                          <button
                            type="button"
                            onClick={clearPendingScans}
                            className="text-sm font-medium text-red-600 transition hover:text-red-700"
                          >
                            Clear all
                          </button>
                        </div>
                        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                          {pendingScans.map((scan) => (
                            <div
                              key={scan.id}
                              className="overflow-hidden rounded-2xl border border-medical-200 bg-medical-50"
                            >
                              <img
                                src={scan.previewUrl}
                                alt={scan.file.name}
                                className="h-28 w-full object-cover"
                              />
                              <div className="space-y-2 p-3">
                                <div className="truncate text-xs font-medium text-medical-700">
                                  {scan.file.name}
                                </div>
                                <button
                                  type="button"
                                  onClick={() => removePendingScan(scan.id)}
                                  className="w-full rounded-xl border border-medical-200 bg-white px-3 py-2 text-xs font-medium text-medical-700 transition hover:border-red-200 hover:text-red-600"
                                >
                                  Remove
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                      Volume calculation is currently mocked, but every result is stored with the study date and scan set.
                    </div>

                    <button
                      type="submit"
                      disabled={creatingStudy || pendingScans.length < 3}
                      className="flex w-full items-center justify-center gap-2 rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-dark disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {creatingStudy ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <ScanLine size={16} />
                      )}
                      Save study and analyze scans
                    </button>
                  </form>
                </div>
              </div>

              <VolumeTrendChart points={trendPoints} />

              <div className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
                <div className="mb-5 flex items-center justify-between gap-4">
                  <div>
                    <h2 className="text-lg font-semibold text-medical-900">
                      Saved studies
                    </h2>
                    <p className="text-sm text-medical-500">
                      Browse historical exams for this patient.
                    </p>
                  </div>
                  {workspaceLoading && (
                    <div className="flex items-center gap-2 text-sm text-medical-500">
                      <Loader2 size={16} className="animate-spin" />
                      Refreshing...
                    </div>
                  )}
                </div>

                {studies.length > 0 ? (
                  <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
                    {studies.map((study) => (
                      <button
                        key={study.id}
                        onClick={() => setSelectedStudyId(study.id)}
                        className={`rounded-3xl border p-5 text-left transition ${
                          selectedStudyId === study.id
                            ? "border-accent bg-accent/10 shadow-sm"
                            : "border-medical-200 bg-medical-50 hover:border-medical-300 hover:bg-white"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="rounded-full border border-medical-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-medical-600">
                            {formatModelName(study.selected_model)}
                          </span>
                          <span className="text-sm font-semibold text-medical-900">
                            {study.volume_mm3.toFixed(2)} mm³
                          </span>
                        </div>

                        <div className="mt-4 text-lg font-semibold text-medical-900">
                          {formatDate(study.study_date)}
                        </div>
                        <div className="mt-2 flex items-center gap-2 text-sm text-medical-500">
                          <CalendarDays size={14} />
                          Saved {formatDateTime(study.created_at)}
                        </div>
                        <div className="mt-4 text-sm text-medical-600">
                          {study.scan_count} scan{study.scan_count === 1 ? "" : "s"} stored
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-3xl border border-dashed border-medical-200 bg-medical-50 px-4 py-10 text-center text-sm text-medical-500">
                    No studies saved for this patient yet.
                  </div>
                )}
              </div>

              <div className="rounded-3xl border border-medical-200 bg-white p-6 shadow-sm">
                <div className="mb-6 flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-lg font-semibold text-medical-900">
                      Study details
                    </h2>
                    <p className="text-sm text-medical-500">
                      Saved scans, segmentation masks and tumor volume for the selected exam.
                    </p>
                  </div>
                  {studyLoading && (
                    <div className="flex items-center gap-2 text-sm text-medical-500">
                      <Loader2 size={16} className="animate-spin" />
                      Loading study...
                    </div>
                  )}
                </div>

                {selectedStudy ? (
                  <div className="space-y-6">
                    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                        <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                          Study date
                        </div>
                        <div className="mt-2 text-lg font-semibold text-medical-900">
                          {formatDate(selectedStudy.study_date)}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                        <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                          Volume
                        </div>
                        <div className="mt-2 text-lg font-semibold text-medical-900">
                          {selectedStudy.volume_mm3.toFixed(2)} mm³
                        </div>
                      </div>
                      <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                        <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                          Model
                        </div>
                        <div className="mt-2 text-lg font-semibold text-medical-900">
                          {formatModelName(selectedStudy.selected_model)}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-medical-200 bg-medical-50 p-4">
                        <div className="text-xs uppercase tracking-[0.2em] text-medical-500">
                          Tumor detections
                        </div>
                        <div className="mt-2 text-lg font-semibold text-medical-900">
                          {selectedStudyTumorCount}
                        </div>
                      </div>
                    </div>

                    {selectedStudy.scans.length > 0 ? (
                      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        {selectedStudy.scans.map((scan) => (
                          <div
                            key={scan.id}
                            className="overflow-hidden rounded-3xl border border-medical-200 bg-medical-50"
                          >
                            <img
                              src={buildApiUrl(scan.image_url)}
                              alt={scan.filename}
                              className="h-48 w-full object-cover"
                            />
                            <div className="space-y-3 p-4">
                              <div>
                                <div className="truncate text-base font-semibold text-medical-900">
                                  {scan.filename}
                                </div>
                                <div className="mt-1 text-sm text-medical-500">
                                  {scan.detections.length} objects detected
                                </div>
                              </div>
                              <div className="flex items-center justify-between gap-3 text-xs text-medical-500">
                                <span>Saved {formatDateTime(scan.created_at)}</span>
                                <span>#{scan.sort_order + 1}</span>
                              </div>
                              <button
                                type="button"
                                onClick={() => setViewerScan(scan)}
                                className="w-full rounded-2xl bg-medical-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-medical-800"
                              >
                                Open segmentation viewer
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="rounded-3xl border border-dashed border-medical-200 bg-medical-50 px-4 py-10 text-center text-sm text-medical-500">
                        This study does not contain scans yet.
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="rounded-3xl border border-dashed border-medical-200 bg-medical-50 px-4 py-10 text-center text-sm text-medical-500">
                    Select a study to inspect its saved scans and segmentation masks.
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex min-h-[560px] items-center justify-center rounded-3xl border border-dashed border-medical-200 bg-white p-10 text-center shadow-sm">
              <div className="max-w-md space-y-4">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-medical-100 text-medical-800">
                  <UserRound size={24} />
                </div>
                <h2 className="text-2xl font-semibold text-medical-900">
                  No patient selected
                </h2>
                <p className="text-sm text-medical-500">
                  Create a patient record in the left panel to start storing OCT studies, masks and tumor volume history.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </Layout>
  );
}

export default App;
