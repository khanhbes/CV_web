import React from 'react';
import { motion } from 'motion/react';
import { ArrowRight, ShieldCheck, Zap, Layers } from 'lucide-react';

export const Introduction: React.FC = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-5xl mx-auto space-y-12 pb-20"
    >
      <header className="space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-600/10 border border-blue-500/20 text-blue-400 text-xs font-medium tracking-wide uppercase">
          <Layers size={12} /> Research Paper 2026
        </div>
        <h2 className="text-5xl font-bold text-white tracking-tight leading-[1.1]">
          Nhận diện & Phân vùng <br />
          <span className="text-blue-500">Đối tượng Vi phạm Giao thông</span>
        </h2>
        <p className="text-xl text-slate-400 leading-relaxed max-w-3xl">
          Nghiên cứu ứng dụng Instance Segmentation đa lớp để tự động hóa quy trình giám sát giao thông thông minh thông qua so sánh thực nghiệm giữa kiến trúc YOLOv26s-seg và RF-DETR.
        </p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="p-8 rounded-2xl bg-slate-900/50 border border-slate-800 hover:border-blue-500/30 transition-colors group">
          <Zap className="text-blue-500 mb-6 group-hover:scale-110 transition-transform" size={40} />
          <h3 className="text-2xl font-bold text-white mb-4">YOLOv26s-seg</h3>
          <p className="text-slate-400 leading-relaxed mb-6">
            Kiến trúc One-stage hiện đại nhất tập trung vào việc tối ưu hóa tốc độ xử lý mà không hy sinh quá nhiều độ chính xác. 
            Sử dụng Cross-Stage Partial Network (CSP) và Path Aggregation Network (PAN) để trích xuất đặc trưng hiệu quả.
          </p>
          <ul className="space-y-3 text-slate-500 text-sm">
            <li className="flex items-center gap-2"><ArrowRight size={14} className="text-blue-500" /> Tốc độ xử lý Real-time (30+ FPS)</li>
            <li className="flex items-center gap-2"><ArrowRight size={14} className="text-blue-500" /> Nhẹ, dễ dàng deploy trên thiết bị nhúng</li>
            <li className="flex items-center gap-2"><ArrowRight size={14} className="text-blue-500" /> Phù hợp cho giám sát liên tục</li>
          </ul>
        </div>

        <div className="p-8 rounded-2xl bg-slate-900/50 border border-slate-800 hover:border-blue-500/30 transition-colors group">
          <ShieldCheck className="text-blue-400 mb-6 group-hover:scale-110 transition-transform" size={40} />
          <h3 className="text-2xl font-bold text-white mb-4">RF-DETR Small</h3>
          <p className="text-slate-400 leading-relaxed mb-6">
            Kiến trúc transformer hiện đại của RF-DETR, dùng DINOv2 backbone và checkpoint RF-DETR_Small.pt để trả về box, mask, confidence cho từng đối tượng.
          </p>
          <ul className="space-y-3 text-slate-500 text-sm">
            <li className="flex items-center gap-2"><ArrowRight size={14} className="text-blue-400" /> Nạp trực tiếp checkpoint RF-DETR_Small.pt</li>
            <li className="flex items-center gap-2"><ArrowRight size={14} className="text-blue-400" /> Hỗ trợ instance segmentation cho ảnh tĩnh</li>
            <li className="flex items-center gap-2"><ArrowRight size={14} className="text-blue-400" /> Phù hợp so sánh với YOLO trong dashboard</li>
          </ul>
        </div>
      </section>

      <section className="bg-slate-900/30 border border-slate-800 rounded-3xl p-10 mt-12">
        <h3 className="text-2xl font-bold text-white mb-8">Dataset Characteristics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { label: 'Tổng số ảnh', value: '5,420', unit: 'IMG' },
            { label: 'Số lượng Classes', value: '4', unit: 'LỚP' },
            { label: 'Annotations', value: '18.4k', unit: 'POLY' },
            { label: 'Độ phân giải', value: '1080p', unit: 'FULLHD' },
          ].map((stat, i) => (
            <div key={i} className="space-y-2 border-l border-slate-800 pl-6">
              <p className="text-xs font-mono text-slate-500 uppercase tracking-widest">{stat.label}</p>
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold text-white">{stat.value}</span>
                <span className="text-[10px] text-slate-600 font-semibold">{stat.unit}</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </motion.div>
  );
};
