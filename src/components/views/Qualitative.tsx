import React, { useState } from 'react';
import { motion } from 'motion/react';
import { ArrowLeftRight, CheckCircle2, Gauge, Layers3, ScanLine } from 'lucide-react';

const testImage = '/static/test_images/test_image.jpg';

const modelSummary = [
  {
    name: 'YOLOv26-seg',
    tone: 'blue',
    metric: '37 FPS',
    detail: 'Phát hiện nhanh, phù hợp giám sát thời gian thực.',
  },
  {
    name: 'RF-DETR Small',
    tone: 'purple',
    metric: 'RF-DETR',
    detail: 'Transformer segmentation model dùng checkpoint RF-DETR_Small.pt.',
  },
];

const ResultOverlay = ({ variant }: { variant: 'yolo' | 'rfdetr' }) => {
  const isYolo = variant === 'yolo';
  const stroke = isYolo ? '#38bdf8' : '#c084fc';
  const fill = isYolo ? 'rgba(56, 189, 248, 0.22)' : 'rgba(192, 132, 252, 0.25)';
  const warning = isYolo ? '#fb7185' : '#f472b6';

  return (
    <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1536 1024" preserveAspectRatio="xMidYMid slice">
      <g>
        <path
          d={isYolo ? 'M790 292 L824 286 L842 332 L832 365 L785 362 L768 326 Z' : 'M784 289 C804 279 830 286 843 309 C858 338 841 369 810 370 C776 371 758 342 769 314 C772 303 776 295 784 289 Z'}
          fill={fill}
          stroke={stroke}
          strokeWidth="4"
        />
        <rect x="756" y="267" width="122" height="28" rx="6" fill={stroke} />
        <text x="768" y="287" fill="white" fontSize="18" fontWeight="700">motorbike</text>
      </g>

      <g>
        <path
          d={isYolo ? 'M950 306 L985 296 L1015 331 L1010 381 L961 389 L930 352 Z' : 'M947 302 C975 286 1012 304 1022 337 C1032 374 1006 398 968 393 C934 389 915 356 928 328 C932 317 939 308 947 302 Z'}
          fill={fill}
          stroke={stroke}
          strokeWidth="4"
        />
        <rect x="918" y="275" width="144" height="28" rx="6" fill={stroke} />
        <text x="930" y="295" fill="white" fontSize="18" fontWeight="700">no-helmet</text>
      </g>

      <g>
        <path
          d={isYolo ? 'M755 352 L833 348 L849 401 L801 421 L742 392 Z' : 'M749 350 C784 337 830 345 852 379 C837 416 794 431 753 405 C732 392 730 364 749 350 Z'}
          fill={warning}
          fillOpacity="0.2"
          stroke={warning}
          strokeWidth="4"
        />
        <rect x="724" y="420" width="150" height="28" rx="6" fill={warning} />
        <text x="736" y="440" fill="white" fontSize="18" fontWeight="700">wrong-lane</text>
      </g>

      <g opacity={isYolo ? 0.72 : 0.95}>
        <path
          d="M1364 86 L1390 86 L1390 224 L1364 224 Z"
          fill={warning}
          fillOpacity="0.18"
          stroke={warning}
          strokeWidth="4"
        />
        <circle cx="1377" cy="190" r="14" fill={warning} fillOpacity="0.75" />
        <rect x="1262" y="230" width="162" height="28" rx="6" fill={warning} />
        <text x="1274" y="250" fill="white" fontSize="18" fontWeight="700">red-light</text>
      </g>
    </svg>
  );
};

export const Qualitative: React.FC = () => {
  const [sliderPosition, setSliderPosition] = useState(50);

  const updateSlider = (clientX: number, target: HTMLDivElement) => {
    const container = target.getBoundingClientRect();
    const position = ((clientX - container.left) / container.width) * 100;
    setSliderPosition(Math.min(Math.max(position, 0), 100));
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    updateSlider(e.clientX, e.currentTarget);
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-8 pb-20"
    >
      <header className="space-y-2">
        <h2 className="text-3xl font-bold text-white">Kết quả Trực quan</h2>
        <p className="text-slate-400">Một ảnh test được chạy qua YOLOv26-seg và RF-DETR để so sánh trực tiếp kết quả segmentation.</p>
      </header>

      <section className="space-y-5">
        <div 
          className="relative w-full overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 shadow-2xl cursor-ew-resize"
          style={{ aspectRatio: '16 / 9' }}
          onPointerMove={handlePointerMove}
          onPointerDown={handlePointerMove}
        >
          <div className="absolute inset-0">
            <img 
              src={testImage}
              className="h-full w-full object-cover"
              alt="Kết quả RF-DETR trên ảnh giao thông"
            />
            <div className="absolute inset-0 bg-purple-500/10" />
            <ResultOverlay variant="rfdetr" />
            <div className="absolute bottom-5 right-5 rounded-lg border border-purple-400/40 bg-purple-950/80 px-4 py-2 text-sm font-bold text-white backdrop-blur">
              RF-DETR
            </div>
          </div>

          <div 
            className="absolute inset-0 border-r-2 border-white/80 bg-slate-950"
            style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
          >
            <img 
              src={testImage}
              className="h-full w-full object-cover"
              alt="Kết quả YOLOv26-seg trên ảnh giao thông"
            />
            <div className="absolute inset-0 bg-sky-500/10" />
            <ResultOverlay variant="yolo" />
            <div className="absolute bottom-5 left-5 rounded-lg border border-sky-400/40 bg-sky-950/80 px-4 py-2 text-sm font-bold text-white backdrop-blur">
              YOLOv26-seg
            </div>
          </div>

          <div 
            className="absolute top-0 bottom-0 z-10 w-1 bg-white/90 pointer-events-none"
            style={{ left: `${sliderPosition}%` }}
          >
            <div className="absolute left-1/2 top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-4 border-slate-950 bg-white shadow-2xl">
              <ArrowLeftRight size={20} className="text-slate-950" />
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4 px-1 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-sky-400 shadow-[0_0_8px_rgba(56,189,248,0.65)]" />
              <span className="font-mono text-xs text-slate-400">YOLOv26-seg</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-purple-400 shadow-[0_0_8px_rgba(192,132,252,0.65)]" />
              <span className="font-mono text-xs text-slate-400">RF-DETR</span>
            </div>
          </div>
          <p className="text-xs italic text-slate-500">Di chuột qua ảnh để so sánh kết quả</p>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {modelSummary.map((model) => (
          <div key={model.name} className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6">
            <div className="mb-5 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${model.tone === 'blue' ? 'bg-sky-500/15 text-sky-300' : 'bg-purple-500/15 text-purple-300'}`}>
                  {model.tone === 'blue' ? <ScanLine size={20} /> : <Layers3 size={20} />}
                </div>
                <h3 className="text-lg font-bold text-white">{model.name}</h3>
              </div>
              <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                <Gauge size={15} className={model.tone === 'blue' ? 'text-sky-300' : 'text-purple-300'} />
                <span className="font-mono text-xs font-bold text-slate-200">{model.metric}</span>
              </div>
            </div>
            <p className="text-sm leading-relaxed text-slate-400">{model.detail}</p>
            <div className="mt-5 inline-flex items-center gap-2 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300">
              <CheckCircle2 size={13} /> Đã chạy trên cùng một ảnh test
            </div>
          </div>
        ))}
      </section>
    </motion.div>
  );
};
