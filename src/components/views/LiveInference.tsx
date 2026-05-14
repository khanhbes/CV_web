import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, Cpu, Sliders, Activity, Info, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { cn } from '../../lib/utils';

export const LiveInference: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState<'yolo' | 'rfdetr'>('yolo');
  const [conf, setConf] = useState(0.45);
  const [iou, setIou] = useState(0.5);
  const [isInferring, setIsInferring] = useState(false);
  const [result, setResult] = useState<boolean>(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(false);
    }
  };

  const runInference = () => {
    setIsInferring(true);
    setTimeout(() => {
      setIsInferring(false);
      setResult(true);
    }, 1500);
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid grid-cols-1 lg:grid-cols-12 gap-8 pb-20"
    >
      {/* Settings Column */}
      <aside className="lg:col-span-4 space-y-6">
        <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-8">
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Cpu size={18} className="text-blue-500" />
              <span>Model Configuration</span>
            </div>
            
            <div className="space-y-3">
              <label className="text-xs text-slate-500 font-mono uppercase tracking-wider">Select Architecture</label>
              <div className="grid grid-cols-2 gap-2">
                <button 
                  onClick={() => setModel('yolo')}
                  className={cn(
                    "px-4 py-2 rounded-lg text-xs font-bold transition-all border",
                    model === 'yolo' ? "bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/20" : "bg-slate-950 border-slate-800 text-slate-500 hover:border-slate-700"
                  )}
                >
                  YOLOv26s-seg
                </button>
                <button 
                  onClick={() => setModel('rfdetr')}
                  className={cn(
                    "px-4 py-2 rounded-lg text-xs font-bold transition-all border",
                    model === 'rfdetr' ? "bg-purple-600 border-purple-500 text-white shadow-lg shadow-purple-900/20" : "bg-slate-950 border-slate-800 text-slate-500 hover:border-slate-700"
                  )}
                >
                  RF-DETR
                </button>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Sliders size={18} className="text-blue-500" />
              <span>Hyperparameters</span>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-xs text-slate-400 font-medium">Confidence Thresh</label>
                  <span className="text-xs text-blue-400 font-mono font-bold">{conf.toFixed(2)}</span>
                </div>
                <input 
                  type="range" min="0" max="1" step="0.01" 
                  value={conf} onChange={(e) => setConf(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-xs text-slate-400 font-medium">IoU Threshold</label>
                  <span className="text-xs text-blue-400 font-mono font-bold">{iou.toFixed(2)}</span>
                </div>
                <input 
                  type="range" min="0" max="1" step="0.01" 
                  value={iou} onChange={(e) => setIou(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>
            </div>
          </div>

          <button 
            disabled={!file || isInferring}
            onClick={runInference}
            className={cn(
              "w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all",
              !file || isInferring 
                ? "bg-slate-800 text-slate-600 cursor-not-allowed" 
                : "bg-blue-600 hover:bg-blue-500 text-white shadow-xl shadow-blue-900/30"
            )}
          >
            {isInferring ? (
              <>
                <RefreshCw className="animate-spin" size={20} />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Activity size={20} />
                <span>Run Inference</span>
              </>
            )}
          </button>
        </div>

        <div className="bg-blue-600/5 border border-blue-500/10 rounded-2xl p-6 flex gap-4">
          <Info className="text-blue-500 shrink-0" size={20} />
          <p className="text-xs text-slate-400 leading-relaxed">
            Hệ thống sẽ tự động vẽ bounding box, mask segmentation và tính toán độ tin cậy cho từng đối tượng trong khung hình.
          </p>
        </div>
      </aside>

      {/* Main Display Column */}
      <main className="lg:col-span-8 space-y-6">
        <div 
          onClick={() => !file && fileInputRef.current?.click()}
          className={cn(
            "relative min-h-[480px] w-full rounded-3xl border-2 border-dashed transition-all flex flex-col items-center justify-center p-4",
            !file 
              ? "border-slate-800 bg-slate-950 hover:border-blue-500/50 hover:bg-slate-900/40 cursor-pointer" 
              : "border-slate-800 bg-slate-900 cursor-default"
          )}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            onChange={handleFileUpload}
            accept="image/*"
          />

          {!preview ? (
            <div className="text-center space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-500">
                <Upload size={30} />
              </div>
              <div>
                <p className="text-white font-bold text-lg">Tải lên hình ảnh nghiên cứu</p>
                <p className="text-slate-500 text-sm">PNG, JPG or JPEG (Max. 10MB)</p>
              </div>
            </div>
          ) : (
            <div className="relative w-full h-full rounded-2xl overflow-hidden shadow-2xl">
              <img src={preview} className="w-full h-full object-contain" alt="Preview" />
              
              {/* Fake Result Overlay */}
              <AnimatePresence>
                {result && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="absolute inset-0 pointer-events-none"
                  >
                    {/* SVG Results Layer */}
                    <svg className="w-full h-full" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid slice">
                      {/* Detection 1 */}
                      <g>
                        <rect x="250" y="300" width="300" height="400" className="fill-blue-500/30 stroke-blue-500 stroke-[4px]" />
                        <path d="M250,300 C300,320 400,310 550,300 L550,700 C450,680 350,710 250,700 Z" className="fill-blue-400/20" />
                        <rect x="250" y="270" width="180" height="30" className="fill-blue-500" />
                        <text x="255" y="291" className="fill-white font-mono text-[18px] font-bold">No-Helmet: {(conf + 0.1).toFixed(2)}</text>
                      </g>
                      
                       {/* Detection 2 */}
                       <g>
                        <rect x="620" y="450" width="120" height="150" className="fill-red-500/30 stroke-red-500 stroke-[4px]" />
                        <rect x="620" y="420" width="140" height="30" className="fill-red-500" />
                        <text x="625" y="441" className="fill-white font-mono text-[18px] font-bold">Wrong-Lane: 0.88</text>
                      </g>
                    </svg>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Status Pill */}
              <div className="absolute top-6 right-6 flex items-center gap-2">
                <button 
                  onClick={() => { setPreview(null); setFile(null); setResult(false); }}
                  className="bg-black/60 backdrop-blur text-white p-2 rounded-full hover:bg-red-500 transition-colors pointer-events-auto"
                >
                  <RefreshCw size={16} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Inference Progress / Report */}
        <AnimatePresence>
          {result && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="bg-slate-900 border border-slate-800 rounded-3xl p-8"
            >
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-500 flex items-center justify-center">
                    <CheckCircle2 size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">Inference Completed</h3>
                    <p className="text-slate-400 text-sm">Detected 2 violations in 28ms</p>
                  </div>
                </div>
                
                <div className="flex gap-4">
                  <div className="px-5 py-3 rounded-2xl bg-slate-950 border border-slate-800">
                    <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">FPS</p>
                    <p className="text-xl font-mono font-bold text-emerald-400">35.2</p>
                  </div>
                  <div className="px-5 py-3 rounded-2xl bg-slate-950 border border-slate-800">
                    <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">Time</p>
                    <p className="text-xl font-mono font-bold text-blue-400">28ms</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </motion.div>
  );
};
