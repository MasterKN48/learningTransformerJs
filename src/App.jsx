import { useState, useRef, useEffect, useCallback } from "react";
import { pipeline, env } from "@huggingface/transformers";
import {
  Upload,
  ImageIcon,
  Loader2,
  Target,
  BrainCircuit,
  Github,
  Settings,
  ShieldCheck,
  Cpu,
  Zap,
  Info,
  Database,
} from "lucide-react";

// Default configuration
const DEFAULT_MODEL =
  import.meta.env.VITE_MODEL_NAME || "onnx-community/rfdetr_medium-ONNX";
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;

function App() {
  // State for Pipeline
  const [detector, setDetector] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [results, setResults] = useState([]);
  const [isReady, setIsReady] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [status, setStatus] = useState("System standby");

  // State for Config
  const [showConfig, setShowConfig] = useState(false);
  const [device, setDevice] = useState("auto"); // auto, webgpu, wasm, cpu
  const [quantized, setQuantized] = useState(true);
  const [modelName, setModelName] = useState(DEFAULT_MODEL);
  const [capabilities, setCapabilities] = useState({
    webgpu: false,
    wasm: true, // Always true in modern browsers
    webnn: false,
    cpu: true,
  });
  const [modelInput, setModelInput] = useState(DEFAULT_MODEL);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Check hardware capabilities
  useEffect(() => {
    async function checkCapabilities() {
      const caps = { webgpu: false, wasm: true, webnn: false, cpu: true };

      // Check WebGPU
      if ("gpu" in navigator) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) caps.webgpu = true;
        } catch (e) {
          console.error("WebGPU not available", e);
        }
      }

      // Check WebNN
      if ("ml" in navigator) {
        caps.webnn = true;
      }

      setCapabilities(caps);
    }
    checkCapabilities();
  }, []);

  // Configure Transformers.js environment
  useEffect(() => {
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    // Note: Local storage location is managed by the browser's Cache API in Transformers.js by default
    // but we can simulate/explain the "Cache Location" in the UI.
  }, []);

  const loadModel = useCallback(async () => {
    setIsReady(false);
    setDetector(null);
    setStatus("Initializing model...");

    try {
      setStatus(`Loading ${modelName} on ${device}...`);

      const options = {
        device: device === "auto" ? undefined : device,
        quantized: quantized,
      };

      // Special handling for WebNN (experimental/browser specific)
      if (device === "webnn") {
        options.device = "webnn";
      }

      const pipe = await pipeline("object-detection", modelName, options);

      setDetector(() => pipe);
      setIsReady(true);
      setStatus("Engine Ready");
    } catch (err) {
      console.error("Initialization error:", err);
      setStatus(`Error: ${err.message}`);
    }
  }, [modelName, device, quantized]);

  // Load model when settings change
  useEffect(() => {
    loadModel();
  }, [loadModel]);

  const handleFile = (file) => {
    setError(null);
    if (file) {
      if (file.size > MAX_FILE_SIZE) {
        setError(`File is too large. Max limit is ${MAX_FILE_SIZE_MB}MB.`);
        return;
      }
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setResults([]);
    }
  };

  const handleFileChange = (e) => {
    handleFile(e.target.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const detectObjects = async () => {
    if (!detector || !imageUrl) return;

    setIsDetecting(true);
    setStatus("Analyzing pixels...");

    try {
      const output = await detector(imageUrl, { threshold: 0.75 });
      setResults(output);
      setStatus(`Analysis complete: ${output.length} objects found`);
    } catch (err) {
      console.error("Detection error:", err);
      setStatus("Detection failed.");
    } finally {
      setIsDetecting(false);
    }
  };

  // Draw bounding boxes
  useEffect(() => {
    if (imageRef.current && results.length > 0) {
      const canvas = canvasRef.current;
      const displayImage = imageRef.current;
      canvas.width = displayImage.clientWidth;
      canvas.height = displayImage.clientHeight;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      results.forEach((prediction) => {
        const { box, label, score } = prediction;
        const { xmin, ymin, xmax, ymax } = box;
        const scaleX = canvas.width;
        const scaleY = canvas.height;
        const x = xmin * scaleX;
        const y = ymin * scaleY;
        const w = (xmax - xmin) * scaleX;
        const h = (ymax - ymin) * scaleY;

        ctx.strokeStyle = "#818cf8";
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "#818cf8";
        const labelText = `${label} ${(score * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(x, y - 25, textWidth + 12, 25);

        ctx.fillStyle = "white";
        ctx.font = "bold 13px Inter, sans-serif";
        ctx.fillText(labelText, x + 6, y - 8);
      });
    }
  }, [results, imageUrl]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      {/* Navigation */}
      <nav className="flex justify-between items-center mb-12">
        <div className="flex items-center gap-4">
          <div className="p-2.5 bg-indigo-600 rounded-2xl shadow-lg shadow-indigo-500/20">
            <BrainCircuit className="text-white" size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Transformer.<span className="text-indigo-500">Vision</span>
            </h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500 font-bold">
              Edge Intelligence
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="p-3 glass-panel hover:bg-white/10 transition-all rounded-xl flex items-center gap-2 text-sm font-medium"
          >
            <Settings
              size={18}
              className={showConfig ? "animate-spin-slow" : ""}
            />
            Config
          </button>
          <a
            href="https://github.com"
            className="p-3 glass-panel hover:bg-white/10 transition-all rounded-xl"
          >
            <Github size={18} />
          </a>
        </div>
      </nav>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Controls & Config */}
        <div className="lg:col-span-4 space-y-6">
          {/* Config Panel */}
          {showConfig && (
            <div className="glass-panel p-6 border-indigo-500/30 bg-indigo-500/5 animate-in fade-in slide-in-from-top-4 duration-300">
              <div className="flex items-center gap-2 mb-6">
                <Settings size={18} className="text-indigo-400" />
                <h2 className="font-bold">System Configuration</h2>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">
                    Execution Provider
                  </label>
                  <select
                    value={device}
                    onChange={(e) => setDevice(e.target.value)}
                    className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:border-indigo-500 outline-none appearance-none cursor-pointer"
                  >
                    <option value="auto">Auto Select</option>
                    <option value="webgpu" disabled={!capabilities.webgpu}>
                      WebGPU{" "}
                      {capabilities.webgpu ? "(Available)" : "(Unsupported)"}
                    </option>
                    <option value="wasm">WASM (Available)</option>
                    <option value="webnn" disabled={!capabilities.webnn}>
                      WebNN{" "}
                      {capabilities.webnn ? "(Available)" : "(Unsupported)"}
                    </option>
                    <option value="cpu">CPU (Available)</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">
                    Model Identifier
                  </label>
                  <input
                    type="text"
                    value={modelInput}
                    onChange={(e) => setModelInput(e.target.value)}
                    className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:border-indigo-500 outline-none font-mono"
                    placeholder="e.g. onnx-community/..."
                  />
                  <p className="text-[10px] text-slate-500 mt-1 italic">
                    Default loaded from .env
                  </p>
                </div>

                <div className="flex items-center justify-between p-3 bg-white/5 rounded-xl border border-white/5">
                  <div className="flex items-center gap-2">
                    <Zap size={16} className="text-amber-400" />
                    <span className="text-sm font-medium">
                      8-bit Quantization
                    </span>
                  </div>
                  <button
                    onClick={() => setQuantized(!quantized)}
                    className={`w-10 h-6 rounded-full transition-colors relative ${quantized ? "bg-indigo-600" : "bg-slate-700"}`}
                  >
                    <div
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${quantized ? "left-5" : "left-1"}`}
                    />
                  </button>
                </div>

                <div>
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">
                    Model Directory
                  </label>
                  <div className="flex items-center gap-2 text-xs text-slate-400 bg-black/20 p-3 rounded-lg border border-white/5 font-mono">
                    <Database size={14} />
                    <span>Browser Cache (IndexedDB)</span>
                  </div>
                </div>

                <button
                  onClick={() => {
                    setModelName(modelInput);
                    // The useEffect below will trigger the reload
                  }}
                  className="w-full py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl text-xs font-bold uppercase tracking-widest transition-all"
                >
                  Apply & Reload Model
                </button>
              </div>
            </div>
          )}

          {/* Main Controls */}
          <div className="glass-panel p-8 space-y-6">
            <div className="space-y-4">
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`group relative flex flex-col items-center justify-center border-2 border-dashed rounded-3xl p-10 transition-all duration-500 ${
                  isDragging
                    ? "border-indigo-500 bg-indigo-500/10 scale-[1.02]"
                    : imageUrl
                      ? "border-indigo-500/50 bg-indigo-500/5"
                      : "border-white/10 hover:border-indigo-500/30"
                }`}
              >
                <input
                  type="file"
                  id="upload"
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileChange}
                />
                <label
                  htmlFor="upload"
                  className="flex flex-col items-center cursor-pointer text-center"
                >
                  <div className="w-16 h-16 bg-indigo-500/10 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Upload className="text-indigo-400" size={24} />
                  </div>
                  <span className="font-bold text-lg mb-1">Upload Image</span>
                  <span className="text-xs text-slate-500 px-4">
                    Max size: {MAX_FILE_SIZE_MB}MB
                  </span>
                </label>
              </div>

              {error && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs flex items-center gap-2 animate-in fade-in slide-in-from-top-2">
                  <Info size={14} />
                  <span>{error}</span>
                </div>
              )}

              <button
                onClick={detectObjects}
                disabled={!isReady || !imageUrl || isDetecting}
                className="w-full flex items-center justify-center gap-3 btn-primary py-4 font-bold text-lg"
              >
                {isDetecting ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    <span>Neural Processing...</span>
                  </>
                ) : (
                  <>
                    <Target size={20} />
                    <span>Run Inference</span>
                  </>
                )}
              </button>
            </div>

            <div className="pt-6 border-t border-white/5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div
                    className={`w-2.5 h-2.5 rounded-full shadow-[0_0_10px_rgba(0,0,0,0.5)] ${isReady ? "bg-emerald-500 shadow-emerald-500/50" : "bg-amber-500 animate-pulse shadow-amber-500/50"}`}
                  />
                  <span className="text-xs font-bold uppercase tracking-widest text-slate-400">
                    Engine Status
                  </span>
                </div>
                {isReady && (
                  <ShieldCheck size={16} className="text-emerald-500" />
                )}
              </div>
              <p className="text-sm text-slate-300 bg-black/40 p-3 rounded-xl border border-white/5 font-mono break-all">
                {status}
              </p>
            </div>
          </div>

          {/* Results List */}
          {results.length > 0 && (
            <div className="glass-panel p-6 animate-in fade-in slide-in-from-bottom-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold flex items-center gap-2">
                  <ImageIcon size={18} className="text-indigo-400" />
                  Predictions
                </h3>
                <span className="text-[10px] bg-white/5 px-2 py-1 rounded text-slate-400 uppercase font-bold tracking-tighter">
                  {results.length} Found
                </span>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                {results.map((res, i) => (
                  <div
                    key={i}
                    className="flex justify-between items-center p-3 rounded-xl bg-white/5 border border-white/5 group hover:bg-white/10 transition-colors"
                  >
                    <span className="font-bold capitalize text-slate-300 text-sm">
                      {res.label}
                    </span>
                    <div className="flex items-center gap-3">
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full"
                          style={{ width: `${res.score * 100}%` }}
                        />
                      </div>
                      <span className="text-[11px] font-mono font-bold text-indigo-400 leading-none">
                        {(res.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Viewport & Docs */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="glass-panel overflow-hidden relative min-h-[500px] flex items-center justify-center bg-slate-900/50">
            {imageUrl ? (
              <div className="relative w-full h-full flex items-center justify-center p-6">
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="Source"
                  className="max-w-full max-h-[75vh] rounded-2xl object-contain shadow-2xl ring-1 ring-white/10"
                />
                <canvas
                  ref={canvasRef}
                  className="absolute pointer-events-none"
                  style={{
                    top: imageRef.current?.offsetTop,
                    left: imageRef.current?.offsetLeft,
                    width: imageRef.current?.clientWidth,
                    height: imageRef.current?.clientHeight,
                  }}
                />
              </div>
            ) : (
              <div className="flex flex-col items-center text-center p-12">
                <div className="w-24 h-24 bg-indigo-500/5 rounded-[2.5rem] flex items-center justify-center mb-8 relative">
                  <div className="absolute inset-0 bg-indigo-500/10 blur-2xl rounded-full" />
                  <ImageIcon
                    className="text-slate-700 relative z-10"
                    size={48}
                  />
                </div>
                <h3 className="text-2xl font-bold mb-3 tracking-tight">
                  Stage is Empty
                </h3>
                <p className="text-slate-500 max-w-sm text-sm leading-relaxed">
                  Start your visual analysis by uploading an image into the
                  neural engine.
                </p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass-panel p-5 space-y-3">
              <div className="p-2 bg-blue-500/10 rounded-lg w-fit">
                <Cpu className="text-blue-400" size={20} />
              </div>
              <h4 className="font-bold text-sm">Edge Computing</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Model executes locally via {device.toUpperCase()} avoiding
                latency and server costs.
              </p>
            </div>
            <div className="glass-panel p-5 space-y-3">
              <div className="p-2 bg-emerald-500/10 rounded-lg w-fit">
                <ShieldCheck className="text-emerald-400" size={20} />
              </div>
              <h4 className="font-bold text-sm">Privacy Vault</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Images are never sent to a server. Data stays inside your
                browser environment.
              </p>
            </div>
            <div className="glass-panel p-5 space-y-3">
              <div className="p-2 bg-amber-500/10 rounded-lg w-fit">
                <Info className="text-amber-400" size={20} />
              </div>
              <h4 className="font-bold text-sm">Model Spec</h4>
              <p
                className="text-xs text-slate-500 leading-relaxed overflow-hidden text-ellipsis whitespace-nowrap"
                title={modelName}
              >
                {modelName}
              </p>
            </div>
          </div>
        </div>
      </main>

      <footer className="mt-20 pt-8 border-t border-white/5 flex justify-between items-center text-slate-500 text-[11px] font-bold uppercase tracking-widest">
        <span>&copy; 2026 Transformers.js Labs</span>
        <div className="flex items-center gap-6">
          <a
            href="https://github.com/MasterKN48/learningTransformerJs#readme"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-400 hover:text-white transition-colors"
          >
            Documentation
          </a>
          <a href="#" className="hover:text-indigo-400 transition-colors">
            Documentation
          </a>
          <a href="#" className="hover:text-indigo-400 transition-colors">
            API Status
          </a>
        </div>
      </footer>
    </div>
  );
}

export default App;
