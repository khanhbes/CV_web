import React from 'react';
import { motion } from 'motion/react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell } from 'recharts';

const mAPData = [
  { name: 'Motorbike', yolo: 87.1, rfdetr: 86.6 },
  { name: 'Red Light', yolo: 82.4, rfdetr: 81.9 },
  { name: 'Stop Line', yolo: 86.9, rfdetr: 86.2 },
  { name: 'No Helmet', yolo: 80.5, rfdetr: 79.8 },
];

const fpsData = [
  { model: 'YOLOv26s-seg', mAP: 84.29, fps: 39.3, size: 35 },
  { model: 'RF-DETR Small', mAP: 83.5, fps: 12.0, size: 80 },
];

export const Quantitative: React.FC = () => {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-10 pb-20"
    >
      <header className="space-y-2">
        <h2 className="text-3xl font-bold text-white">So sánh Định lượng</h2>
        <p className="text-slate-400">Phân tích hiệu suất mô hình dựa trên các chỉ số mAP50 và Tốc độ xử lý (FPS).</p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
        {/* mAP Comparison */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-8">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-slate-200">mAP50 per Class (Segmentation)</h3>
            <p className="text-xs text-slate-500">So sánh độ chính xác phân vùng giữa Two-stage và One-stage</p>
          </div>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mAPData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Legend iconType="circle" />
                <Bar dataKey="yolo" name="YOLOv26s-seg" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rfdetr" name="RF-DETR" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Speed vs Accuracy */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-8">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-slate-200">Độ chính xác vs. Tốc độ (FPS)</h3>
            <p className="text-xs text-slate-500">Mối quan hệ giữa thời gian xử lý và mAP</p>
          </div>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="fps" name="Tốc độ (FPS)" unit=" fps" stroke="#94a3b8" axisLine={false} />
                <YAxis type="number" dataKey="mAP" name="mAP50" unit="%" domain={[75, 90]} stroke="#94a3b8" axisLine={false} />
                <ZAxis type="number" dataKey="size" range={[100, 1000]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Mô hình" data={fpsData} fill="#10b981">
                  {fpsData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.model.includes('YOLO') ? '#3b82f6' : '#a855f7'}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/60 font-mono text-[11px] uppercase tracking-wider text-slate-500">
              <th className="px-6 py-4 font-semibold">Model Name</th>
              <th className="px-6 py-4 font-semibold">mAP@50</th>
              <th className="px-6 py-4 font-semibold">Precision</th>
              <th className="px-6 py-4 font-semibold">Recall</th>
              <th className="px-6 py-4 font-semibold">F1</th>
              <th className="px-6 py-4 font-semibold">FPS</th>
            </tr>
          </thead>
          <tbody className="text-slate-300 divide-y divide-slate-800">
            {[
              { name: 'YOLOv26s-seg', map: '86.9%', precision: '87.6%', recall: '81.0%', f1: '84.2%', fps: '39.3' },
              { name: 'RF-DETR Small', map: '83.5%', precision: '89.8%', recall: '80.7%', f1: '85.0%', fps: '12.0' },
            ].map((row, i) => (
              <tr key={i} className="hover:bg-slate-800/30 transition-colors group px-6 py-4">
                <td className="px-6 py-4 font-medium text-white">{row.name}</td>
                <td className="px-6 py-4 font-mono text-blue-400">{row.map}</td>
                <td className="px-6 py-4 font-mono text-violet-400">{row.precision}</td>
                <td className="px-6 py-4 font-mono text-amber-400">{row.recall}</td>
                <td className="px-6 py-4 font-mono text-rose-400">{row.f1}</td>
                <td className="px-6 py-4 text-emerald-500 font-bold">{row.fps}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
};
