import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowUpDown,
  CheckCircle2,
  Database,
  Layers,
  RefreshCw,
  ShieldAlert,
  Upload,
  XCircle,
  Eye
} from 'lucide-react';

interface DiseasePrediction {
  disease: string;
  probability: float;
  threshold: float;
  is_positive: boolean;
  test_auc?: float;
  test_f1?: float;
  test_precision?: float;
  test_recall?: float;
}

type float = number;

interface PredictionResponse {
  predictions: DiseasePrediction[];
  raw_predictions: DiseasePrediction[];
  positive_findings: string[];
  is_no_finding: boolean;
  model_info: {
    architecture: string;
    checkpoint: string;
    device: string;
    classes_count: number;
  };
  disclaimer: string;
}

interface BackendHealth {
  status: string;
  model_loaded: boolean;
  device: string;
  checkpoint: string;
  classes_count: number;
}

const DISEASE_DESCRIPTIONS: Record<string, string> = {
  Atelectasis: "Partial collapse or closure of a lung or lung lobe",
  Cardiomegaly: "Enlargement of the cardiac silhouette",
  Consolidation: "Alveolar spaces filled with fluid or exudate",
  Edema: "Fluid accumulation in lung tissue and air spaces",
  Effusion: "Abnormal collection of fluid in the pleural space",
  Emphysema: "Hyperinflated lungs with destruction of alveolar walls",
  Fibrosis: "Scarring and thickening of lung interstitium",
  Hernia: "Protrusion of abdominal organ into the thoracic cavity",
  Infiltration: "Substance denser than air lingering in lung parenchyma",
  Mass: "Discrete soft-tissue opacity larger than 3 cm",
  "No Finding": "Absence of identifiable thoracic radiologic abnormalities",
  Nodule: "Discrete, round or oval pulmonary opacity ≤ 3 cm",
  Pleural_Thickening: "Thickening or calcification of the pleural membrane",
  Pneumonia: "Inflammatory condition of the lung primarily affecting alveoli",
  Pneumothorax: "Presence of air or gas within the pleural cavity"
};

const API_BASE = "http://localhost:8000";

export const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [backendHealth, setBackendHealth] = useState<BackendHealth | null>(null);
  const [backendChecking, setBackendChecking] = useState<boolean>(false);

  // UI view controls
  const [filterMode, setFilterMode] = useState<"all" | "positive">("all");
  const [sortBy, setSortBy] = useState<"prob" | "auc">("prob");
  const [activeTab, setActiveTab] = useState<"dashboard" | "methodology">("dashboard");
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check backend health on mount
  const checkHealth = async () => {
    setBackendChecking(true);
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendHealth(data);
      } else {
        setBackendHealth(null);
      }
    } catch {
      setBackendHealth(null);
    } finally {
      setBackendChecking(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file (PNG, JPEG, WebP).");
      return;
    }
    if (selectedFile.size > 15 * 1024 * 1024) {
      setError("File size exceeds 15 MB limit.");
      return;
    }
    setError(null);
    setResult(null);
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleRunInference = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Unknown server error" }));
        throw new Error(errData.detail || `Server responded with HTTP ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(
        err.message || "Failed to connect to the RAD-X inference service. Make sure backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Filter and sort predictions
  const displayedPredictions = result?.predictions
    ? [...result.predictions]
        .filter((p) => (filterMode === "positive" ? p.is_positive : true))
        .sort((a, b) => {
          if (sortBy === "prob") {
            return b.probability - a.probability;
          }
          return (b.test_auc ?? 0) - (a.test_auc ?? 0);
        })
    : [];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top Warning Banner */}
      <div
        style={{
          background: "linear-gradient(90deg, #1e1b4b 0%, #172554 100%)",
          borderBottom: "1px solid rgba(147, 197, 253, 0.2)",
          padding: "8px 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: "13px",
          color: "#93c5fd",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <ShieldAlert size={16} color="#38bdf8" />
          <span>
            <strong>RESEARCH & EDUCATIONAL PROTOTYPE ONLY</strong> — Not certified for clinical diagnosis, patient triage, or medical management.
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px", fontSize: "12px" }}>
          <span>Dataset: NIH ChestX-ray14 Subset</span>
          <span>Holdout Test: 2,121 Images</span>
        </div>
      </div>

      {/* Main Navbar */}
      <header
        style={{
          borderBottom: "1px solid var(--border-subtle)",
          background: "rgba(11, 17, 32, 0.8)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 50,
          padding: "14px 28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div
            style={{
              width: "40px",
              height: "40px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 15px rgba(6, 182, 212, 0.4)",
            }}
          >
            <Activity size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <h1 style={{ fontSize: "18px", fontWeight: "800", letterSpacing: "0.5px" }}>RAD-X</h1>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "2px 8px",
                  borderRadius: "20px",
                  background: "rgba(6, 182, 212, 0.15)",
                  color: "#38bdf8",
                  border: "1px solid rgba(6, 182, 212, 0.3)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                DenseNet121 v1.0
              </span>
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>
              Multi-Label Thoracic Disease Classification
            </p>
          </div>
        </div>

        {/* Navigation Tabs & Health Badge */}
        <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
          <div
            style={{
              display: "flex",
              background: "var(--bg-secondary)",
              padding: "4px",
              borderRadius: "10px",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <button
              onClick={() => setActiveTab("dashboard")}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: "600",
                background: activeTab === "dashboard" ? "var(--bg-surface)" : "transparent",
                color: activeTab === "dashboard" ? "#38bdf8" : "var(--text-muted)",
                border: "none",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                transition: "all 0.2s ease",
              }}
            >
              <Eye size={15} />
              Inference Dashboard
            </button>
            <button
              onClick={() => setActiveTab("methodology")}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: "600",
                background: activeTab === "methodology" ? "var(--bg-surface)" : "transparent",
                color: activeTab === "methodology" ? "#38bdf8" : "var(--text-muted)",
                border: "none",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                transition: "all 0.2s ease",
              }}
            >
              <Database size={15} />
              Architecture & Metrics
            </button>
          </div>

          {/* Backend Status Indicator */}
          <div
            onClick={checkHealth}
            title="Click to re-check API connection"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "6px 12px",
              borderRadius: "8px",
              background: backendHealth ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)",
              border: `1px solid ${backendHealth ? "rgba(16, 185, 129, 0.25)" : "rgba(239, 68, 68, 0.25)"}`,
              fontSize: "12px",
              cursor: "pointer",
            }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: backendHealth ? "#10b981" : "#ef4444",
                boxShadow: backendHealth ? "0 0 8px #10b981" : "0 0 8px #ef4444",
              }}
            />
            <span style={{ color: backendHealth ? "#34d399" : "#f87171", fontWeight: "600" }}>
              {backendChecking
                ? "Connecting..."
                : backendHealth
                ? `API Online (${backendHealth.device.toUpperCase()})`
                : "API Offline (Port 8000)"}
            </span>
            <RefreshCw size={12} className={backendChecking ? "pulse-indicator" : ""} color="var(--text-dim)" />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: "28px", maxWidth: "1560px", margin: "0 auto", width: "100%" }}>
        {activeTab === "dashboard" ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.35fr", gap: "28px" }}>
            {/* Left Column: Upload & Radiograph Preview */}
            <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
              {/* Image Input Panel */}
              <div className="glass-panel" style={{ padding: "24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                  <h2 style={{ fontSize: "16px", fontWeight: "700", display: "flex", alignItems: "center", gap: "8px" }}>
                    <Upload size={18} color="#38bdf8" />
                    Input Radiograph
                  </h2>
                  {file && (
                    <button
                      onClick={handleReset}
                      style={{
                        background: "none",
                        border: "none",
                        color: "var(--text-muted)",
                        fontSize: "12px",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        padding: "4px 8px",
                        borderRadius: "4px",
                      }}
                    >
                      <XCircle size={14} /> Clear
                    </button>
                  )}
                </div>

                {!previewUrl ? (
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                      border: `2px dashed ${isDragOver ? "var(--accent-cyan)" : "var(--border-subtle)"}`,
                      borderRadius: "var(--radius-md)",
                      padding: "48px 20px",
                      textAlign: "center",
                      background: isDragOver ? "var(--bg-surface-hover)" : "rgba(15, 23, 42, 0.4)",
                      cursor: "pointer",
                      transition: "all 0.2s ease",
                    }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      style={{ display: "none" }}
                      onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                    />
                    <div
                      style={{
                        width: "56px",
                        height: "56px",
                        borderRadius: "50%",
                        background: "rgba(6, 182, 212, 0.1)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        margin: "0 auto 16px",
                        color: "#38bdf8",
                      }}
                    >
                      <Upload size={28} />
                    </div>
                    <p style={{ fontWeight: "600", fontSize: "15px", marginBottom: "6px" }}>
                      Drop chest radiograph here, or <span style={{ color: "#38bdf8" }}>browse</span>
                    </p>
                    <p style={{ fontSize: "12px", color: "var(--text-dim)" }}>
                      Supports DICOM exports, PNG, JPEG, or WebP up to 15MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <div
                      style={{
                        position: "relative",
                        background: "#000000",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border-subtle)",
                        overflow: "hidden",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        maxHeight: "380px",
                        minHeight: "260px",
                      }}
                    >
                      <img
                        src={previewUrl}
                        alt="Chest X-Ray Preview"
                        style={{
                          maxWidth: "100%",
                          maxHeight: "380px",
                          objectFit: "contain",
                          display: "block",
                        }}
                      />
                      {loading && (
                        <div className="scanner-overlay">
                          <div className="scanner-line" />
                          <div
                            style={{
                              position: "absolute",
                              bottom: "16px",
                              left: "50%",
                              transform: "translateX(-50%)",
                              background: "rgba(11, 17, 32, 0.85)",
                              padding: "6px 16px",
                              borderRadius: "20px",
                              border: "1px solid var(--border-focus)",
                              fontSize: "12px",
                              color: "#38bdf8",
                              fontWeight: "600",
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                            }}
                          >
                            <RefreshCw size={14} className="pulse-indicator" />
                            Evaluating 15 Pathological Targets...
                          </div>
                        </div>
                      )}
                    </div>

                    <div
                      style={{
                        marginTop: "14px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        fontSize: "12px",
                        color: "var(--text-muted)",
                      }}
                    >
                      <span style={{ fontFamily: "var(--font-mono)" }}>
                        {file?.name} ({(file?.size ? file.size / (1024 * 1024) : 0).toFixed(2)} MB)
                      </span>
                      <button
                        onClick={handleRunInference}
                        disabled={loading || !backendHealth}
                        style={{
                          background: loading || !backendHealth
                            ? "var(--bg-surface)"
                            : "linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)",
                          color: loading || !backendHealth ? "var(--text-dim)" : "#ffffff",
                          border: "none",
                          padding: "10px 22px",
                          borderRadius: "8px",
                          fontWeight: "700",
                          fontSize: "13px",
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          boxShadow: loading || !backendHealth ? "none" : "0 0 15px rgba(6, 182, 212, 0.3)",
                        }}
                      >
                        <Activity size={16} />
                        {loading ? "Analyzing..." : "Run Multi-Label Analysis"}
                      </button>
                    </div>
                  </div>
                )}

                {error && (
                  <div
                    style={{
                      marginTop: "16px",
                      padding: "12px 16px",
                      borderRadius: "8px",
                      background: "rgba(239, 68, 68, 0.12)",
                      border: "1px solid rgba(239, 68, 68, 0.3)",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      fontSize: "13px",
                      color: "#fca5a5",
                    }}
                  >
                    <AlertTriangle size={18} color="#f87171" style={{ flexShrink: 0 }} />
                    <span>{error}</span>
                  </div>
                )}
              </div>

              {/* Research Methodology Card */}
              <div className="glass-panel" style={{ padding: "20px" }}>
                <h3 style={{ fontSize: "14px", fontWeight: "700", color: "#38bdf8", marginBottom: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <Layers size={16} />
                  Model Pipeline Specs
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", fontSize: "12px" }}>
                  <div style={{ background: "var(--bg-secondary)", padding: "10px 12px", borderRadius: "8px" }}>
                    <div style={{ color: "var(--text-dim)" }}>Architecture</div>
                    <div style={{ fontWeight: "600", marginTop: "2px" }}>DenseNet121 (ImageNet weights)</div>
                  </div>
                  <div style={{ background: "var(--bg-secondary)", padding: "10px 12px", borderRadius: "8px" }}>
                    <div style={{ color: "var(--text-dim)" }}>Input Resolution</div>
                    <div style={{ fontWeight: "600", marginTop: "2px" }}>224 × 224 (3 Channels)</div>
                  </div>
                  <div style={{ background: "var(--bg-secondary)", padding: "10px 12px", borderRadius: "8px" }}>
                    <div style={{ color: "var(--text-dim)" }}>Class Weighting</div>
                    <div style={{ fontWeight: "600", marginTop: "2px" }}>Pos-Weight capped at 10.0</div>
                  </div>
                  <div style={{ background: "var(--bg-secondary)", padding: "10px 12px", borderRadius: "8px" }}>
                    <div style={{ color: "var(--text-dim)" }}>Patient Partition</div>
                    <div style={{ fontWeight: "600", marginTop: "2px" }}>Zero-Leakage (70/15/15)</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Prediction Results & Calibrated Thresholds */}
            <div>
              {!result && !loading && (
                <div
                  className="glass-panel"
                  style={{
                    padding: "60px 30px",
                    textAlign: "center",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    minHeight: "520px",
                  }}
                >
                  <div
                    style={{
                      width: "64px",
                      height: "64px",
                      borderRadius: "50%",
                      background: "rgba(2, 132, 199, 0.1)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#38bdf8",
                      marginBottom: "16px",
                    }}
                  >
                    <Activity size={32} />
                  </div>
                  <h3 style={{ fontSize: "18px", fontWeight: "700", marginBottom: "8px" }}>
                    Awaiting Image Input
                  </h3>
                  <p style={{ color: "var(--text-muted)", fontSize: "13px", maxWidth: "420px", lineHeight: "1.6" }}>
                    Upload an anterior-posterior (AP) or posterior-anterior (PA) chest radiograph to evaluate disease probabilities across 15 thoracic categories using validation-calibrated thresholds.
                  </p>
                </div>
              )}

              {loading && (
                <div
                  className="glass-panel"
                  style={{
                    padding: "60px 30px",
                    textAlign: "center",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    minHeight: "520px",
                  }}
                >
                  <div
                    style={{
                      width: "64px",
                      height: "64px",
                      borderRadius: "50%",
                      border: "3px solid rgba(6, 182, 212, 0.2)",
                      borderTopColor: "#38bdf8",
                      animation: "spin 1s linear infinite",
                      marginBottom: "20px",
                    }}
                  />
                  <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                  <h3 style={{ fontSize: "16px", fontWeight: "700", marginBottom: "6px" }}>
                    Computing Deep Learning Feature Maps...
                  </h3>
                  <p style={{ color: "var(--text-muted)", fontSize: "13px" }}>
                    Extracting feature activations and applying decision boundaries.
                  </p>
                </div>
              )}

              {result && !loading && (
                <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                  {/* Summary Banner */}
                  <div
                    className="glass-panel"
                    style={{
                      padding: "20px 24px",
                      background: result.positive_findings.length > 0
                        ? "linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(17, 29, 53, 0.8) 100%)"
                        : "linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(17, 29, 53, 0.8) 100%)",
                      border: `1px solid ${result.positive_findings.length > 0 ? "rgba(239, 68, 68, 0.3)" : "rgba(16, 185, 129, 0.3)"}`,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
                          {result.positive_findings.length > 0 ? (
                            <>
                              <AlertTriangle size={20} color="#f87171" />
                              <span style={{ fontSize: "16px", fontWeight: "800", color: "#f87171" }}>
                                {result.positive_findings.length} Pathological Finding{result.positive_findings.length > 1 ? "s" : ""} Above Decision Threshold
                              </span>
                            </>
                          ) : (
                            <>
                              <CheckCircle2 size={20} color="#34d399" />
                              <span style={{ fontSize: "16px", fontWeight: "800", color: "#34d399" }}>
                                No Pathological Findings Above Threshold
                              </span>
                            </>
                          )}
                        </div>
                        <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
                          {result.positive_findings.length > 0
                            ? `Classified positive: ${result.positive_findings.join(", ")}.`
                            : "All 14 thoracic disease categories remained below their validation-tuned decision thresholds."}
                        </p>
                      </div>

                      <div
                        style={{
                          textAlign: "right",
                          fontFamily: "var(--font-mono)",
                          fontSize: "12px",
                          color: "var(--text-dim)",
                        }}
                      >
                        <div>Model: {result.model_info.architecture}</div>
                        <div>Target Classes: {result.model_info.classes_count}</div>
                      </div>
                    </div>
                  </div>

                  {/* Filters & Sorting Toolbar */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "0 4px",
                    }}
                  >
                    <div style={{ display: "flex", gap: "8px" }}>
                      <button
                        onClick={() => setFilterMode("all")}
                        style={{
                          padding: "6px 12px",
                          borderRadius: "6px",
                          fontSize: "12px",
                          fontWeight: "600",
                          background: filterMode === "all" ? "var(--accent-blue-deep)" : "var(--bg-surface)",
                          color: filterMode === "all" ? "#ffffff" : "var(--text-muted)",
                          border: "1px solid var(--border-subtle)",
                        }}
                      >
                        All Classes (15)
                      </button>
                      <button
                        onClick={() => setFilterMode("positive")}
                        style={{
                          padding: "6px 12px",
                          borderRadius: "6px",
                          fontSize: "12px",
                          fontWeight: "600",
                          background: filterMode === "positive" ? "rgba(239, 68, 68, 0.2)" : "var(--bg-surface)",
                          color: filterMode === "positive" ? "#f87171" : "var(--text-muted)",
                          border: `1px solid ${filterMode === "positive" ? "rgba(239, 68, 68, 0.4)" : "var(--border-subtle)"}`,
                        }}
                      >
                        Positive Findings ({result.positive_findings.length})
                      </button>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span style={{ fontSize: "12px", color: "var(--text-dim)", display: "flex", alignItems: "center", gap: "4px" }}>
                        <ArrowUpDown size={12} /> Sort:
                      </span>
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as "prob" | "auc")}
                        style={{
                          background: "var(--bg-surface)",
                          color: "var(--text-main)",
                          border: "1px solid var(--border-subtle)",
                          padding: "4px 8px",
                          borderRadius: "6px",
                          fontSize: "12px",
                          outline: "none",
                        }}
                      >
                        <option value="prob">Highest Probability</option>
                        <option value="auc">Highest Test AUC-ROC</option>
                      </select>
                    </div>
                  </div>

                  {/* Multi-Label Results Chart */}
                  <div className="glass-panel" style={{ padding: "20px" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "11px",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                        color: "var(--text-dim)",
                        paddingBottom: "12px",
                        borderBottom: "1px solid var(--border-subtle)",
                        marginBottom: "14px",
                      }}
                    >
                      <span style={{ flex: 1.4 }}>Disease Target</span>
                      <span style={{ flex: 2.2 }}>Probability vs Validation Boundary</span>
                      <span style={{ flex: 1.1, textAlign: "right" }}>Decision & Test AUC</span>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                      {displayedPredictions.map((pred) => {
                        const probPercent = Math.round(pred.probability * 100);
                        const thresholdPercent = Math.round(pred.threshold * 100);
                        const isPos = pred.is_positive;

                        return (
                          <div
                            key={pred.disease}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "14px",
                              padding: "8px 0",
                            }}
                          >
                            {/* Disease Label & Clinical Subtitle */}
                            <div style={{ flex: 1.4 }}>
                              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                <span style={{ fontWeight: "700", fontSize: "14px" }}>{pred.disease}</span>
                              </div>
                              <div style={{ fontSize: "11px", color: "var(--text-dim)", marginTop: "2px", lineHeight: "1.3" }}>
                                {DISEASE_DESCRIPTIONS[pred.disease] || "Thoracic condition"}
                              </div>
                            </div>

                            {/* Probability Bar with Calibrated Threshold Tick */}
                            <div style={{ flex: 2.2, position: "relative" }}>
                              <div
                                style={{
                                  height: "18px",
                                  background: "var(--bg-secondary)",
                                  borderRadius: "4px",
                                  position: "relative",
                                  overflow: "hidden",
                                  border: "1px solid rgba(255, 255, 255, 0.05)",
                                }}
                              >
                                {/* Fill Bar */}
                                <div
                                  style={{
                                    height: "100%",
                                    width: `${Math.min(100, Math.max(2, probPercent))}%`,
                                    background: isPos
                                      ? "linear-gradient(90deg, #ef4444 0%, #f87171 100%)"
                                      : "linear-gradient(90deg, #0284c7 0%, #38bdf8 100%)",
                                    borderRadius: "4px",
                                    transition: "width 0.6s cubic-bezier(0.16, 1, 0.3, 1)",
                                  }}
                                />

                                {/* Threshold Line Marker */}
                                <div
                                  title={`Decision Threshold: ${thresholdPercent}%`}
                                  style={{
                                    position: "absolute",
                                    left: `${thresholdPercent}%`,
                                    top: 0,
                                    bottom: 0,
                                    width: "2px",
                                    background: "#fbbf24",
                                    zIndex: 2,
                                    boxShadow: "0 0 6px rgba(251, 191, 36, 0.8)",
                                  }}
                                />
                              </div>

                              {/* Numeric labels below bar */}
                              <div
                                style={{
                                  display: "flex",
                                  justifyContent: "space-between",
                                  fontSize: "11px",
                                  fontFamily: "var(--font-mono)",
                                  marginTop: "4px",
                                  color: "var(--text-muted)",
                                }}
                              >
                                <span style={{ color: isPos ? "#f87171" : "#38bdf8", fontWeight: "600" }}>
                                  {(pred.probability * 100).toFixed(1)}% prob
                                </span>
                                <span style={{ color: "#fbbf24", fontSize: "10px" }}>
                                  Threshold: {(pred.threshold * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>

                            {/* Classification Badge & Test AUC */}
                            <div style={{ flex: 1.1, textAlign: "right" }}>
                              <span
                                style={{
                                  display: "inline-block",
                                  padding: "3px 8px",
                                  borderRadius: "4px",
                                  fontSize: "11px",
                                  fontWeight: "700",
                                  fontFamily: "var(--font-mono)",
                                  background: isPos ? "var(--status-positive-bg)" : "var(--status-negative-bg)",
                                  color: isPos ? "var(--status-positive-text)" : "var(--status-negative-text)",
                                  border: `1px solid ${isPos ? "var(--status-positive-border)" : "var(--status-negative-border)"}`,
                                }}
                              >
                                {isPos ? "POSITIVE" : "NEGATIVE"}
                              </span>
                              {pred.test_auc && (
                                <div
                                  style={{
                                    fontSize: "10px",
                                    color: "var(--text-dim)",
                                    fontFamily: "var(--font-mono)",
                                    marginTop: "3px",
                                  }}
                                >
                                  Test AUC: {pred.test_auc.toFixed(3)}
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Methodology & Metrics Tab */
          <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
            <div className="glass-panel" style={{ padding: "28px" }}>
              <h2 style={{ fontSize: "20px", fontWeight: "800", marginBottom: "8px", color: "#38bdf8" }}>
                Experimental Methodology & Validation Setup
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "14px", lineHeight: "1.7", maxWidth: "900px" }}>
                RAD-X implements transfer learning on DenseNet121 for thoracic multi-label classification.
                To eliminate data leakage, the 14,999 chest radiographs across 3,923 patients were split strictly at the patient level
                (70% train / 15% validation / 15% test). No patient's images exist in more than one partition.
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px", marginTop: "24px" }}>
                <div style={{ background: "var(--bg-secondary)", padding: "18px", borderRadius: "12px", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "12px", color: "var(--text-dim)", textTransform: "uppercase" }}>Training Set</div>
                  <div style={{ fontSize: "24px", fontWeight: "800", color: "#38bdf8", marginTop: "4px" }}>10,699</div>
                  <div style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "2px" }}>2,746 Unique Patients (70%)</div>
                </div>
                <div style={{ background: "var(--bg-secondary)", padding: "18px", borderRadius: "12px", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "12px", color: "var(--text-dim)", textTransform: "uppercase" }}>Validation Set</div>
                  <div style={{ fontSize: "24px", fontWeight: "800", color: "#38bdf8", marginTop: "4px" }}>2,179</div>
                  <div style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "2px" }}>588 Patients (Used for threshold tuning)</div>
                </div>
                <div style={{ background: "var(--bg-secondary)", padding: "18px", borderRadius: "12px", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "12px", color: "var(--text-dim)", textTransform: "uppercase" }}>Holdout Test Set</div>
                  <div style={{ fontSize: "24px", fontWeight: "800", color: "#38bdf8", marginTop: "4px" }}>2,121</div>
                  <div style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "2px" }}>589 Patients (Evaluated strictly once)</div>
                </div>
              </div>
            </div>

            {/* Test Performance Table */}
            <div className="glass-panel" style={{ padding: "28px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
                <div>
                  <h3 style={{ fontSize: "16px", fontWeight: "700" }}>Holdout Test Set Metrics by Disease</h3>
                  <p style={{ fontSize: "13px", color: "var(--text-dim)", marginTop: "2px" }}>
                    Performance on untouched 2,121 test radiographs after two-stage transfer learning fine-tuning.
                  </p>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "13px", color: "#38bdf8", background: "rgba(6, 182, 212, 0.1)", padding: "6px 14px", borderRadius: "8px" }}>
                  Mean Test AUC: 0.7383
                </div>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left", color: "var(--text-dim)", fontSize: "11px", textTransform: "uppercase" }}>
                      <th style={{ padding: "10px 14px" }}>Target Disease</th>
                      <th style={{ padding: "10px 14px" }}>AUC-ROC</th>
                      <th style={{ padding: "10px 14px" }}>F1 Score</th>
                      <th style={{ padding: "10px 14px" }}>Precision</th>
                      <th style={{ padding: "10px 14px" }}>Recall</th>
                      <th style={{ padding: "10px 14px" }}>Calibrated Threshold</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { d: "Emphysema", auc: 0.8565, f1: 0.2883, p: 0.1963, r: 0.5424, t: 0.3839 },
                      { d: "Cardiomegaly", auc: 0.8548, f1: 0.2400, p: 0.3750, r: 0.1765, t: 0.6858 },
                      { d: "Edema", auc: 0.8414, f1: 0.0000, p: 0.0000, r: 0.0000, t: 0.5286 },
                      { d: "Effusion", auc: 0.8096, f1: 0.3957, p: 0.3834, r: 0.4088, t: 0.7964 },
                      { d: "Pneumothorax", auc: 0.7762, f1: 0.2555, p: 0.2229, r: 0.2991, t: 0.6157 },
                      { d: "Consolidation", auc: 0.7493, f1: 0.0734, p: 0.1081, r: 0.0556, t: 0.6810 },
                      { d: "Atelectasis", auc: 0.7435, f1: 0.2659, p: 0.1877, r: 0.4562, t: 0.5752 },
                      { d: "Fibrosis", auc: 0.7377, f1: 0.1087, p: 0.1000, r: 0.1190, t: 0.6182 },
                      { d: "Pneumonia", auc: 0.7038, f1: 0.0469, p: 0.0316, r: 0.0909, t: 0.1885 },
                      { d: "No Finding", auc: 0.7016, f1: 0.7478, p: 0.6501, r: 0.8802, t: 0.2685 },
                      { d: "Mass", auc: 0.6999, f1: 0.0787, p: 0.0737, r: 0.0843, t: 0.5274 },
                      { d: "Infiltration", auc: 0.6953, f1: 0.3212, p: 0.2736, r: 0.3889, t: 0.5793 },
                      { d: "Pleural_Thickening", auc: 0.6916, f1: 0.1451, p: 0.1284, r: 0.1667, t: 0.5444 },
                      { d: "Hernia", auc: 0.6420, f1: 0.0000, p: 0.0000, r: 0.0000, t: 0.1571 },
                      { d: "Nodule", auc: 0.6273, f1: 0.1606, p: 0.1399, r: 0.1887, t: 0.4892 },
                    ].map((row, idx) => (
                      <tr
                        key={row.d}
                        style={{
                          borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                          background: idx % 2 === 0 ? "rgba(17, 29, 53, 0.3)" : "transparent",
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        <td style={{ padding: "12px 14px", fontFamily: "var(--font-sans)", fontWeight: "600" }}>{row.d}</td>
                        <td style={{ padding: "12px 14px", color: row.auc >= 0.8 ? "#34d399" : row.auc >= 0.7 ? "#38bdf8" : "var(--text-muted)" }}>
                          {row.auc.toFixed(4)}
                        </td>
                        <td style={{ padding: "12px 14px" }}>{row.f1.toFixed(4)}</td>
                        <td style={{ padding: "12px 14px" }}>{row.p.toFixed(4)}</td>
                        <td style={{ padding: "12px 14px" }}>{row.r.toFixed(4)}</td>
                        <td style={{ padding: "12px 14px", color: "#fbbf24" }}>{row.t.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "20px 28px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "12px",
          color: "var(--text-dim)",
          background: "var(--bg-secondary)",
        }}
      >
        <div>
          RAD-X Diagnostic Prototype • DenseNet121 • PyTorch & FastAPI
        </div>
        <div>
          Strictly for educational and research evaluation. Not for clinical use.
        </div>
      </footer>
    </div>
  );
};

export default App;
