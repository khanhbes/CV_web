import React from 'react';
import { Home, BarChart2, Image as ImageIcon, Play, Cpu, Database, Menu, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '../lib/utils';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}

const menuItems = [
  { id: 'introduction', label: 'Giới thiệu', icon: Home },
  { id: 'quantitative', label: 'Định lượng', icon: BarChart2 },
  { id: 'qualitative', label: 'Trực quan', icon: ImageIcon },
  { id: 'inference', label: 'Demo Thực tế', icon: Play },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, isOpen, setIsOpen }) => {
  return (
    <>
      {/* Mobile Toggle */}
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-slate-900 text-white rounded-md border border-slate-800"
      >
        {isOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Sidebar Content */}
      <motion.aside
        initial={false}
        animate={{ x: isOpen ? 0 : -280 }}
        className={cn(
          "fixed top-0 left-0 h-full w-[280px] bg-slate-950 border-r border-slate-800 z-40 flex flex-col transition-all duration-300 ease-in-out",
          !isOpen && "lg:translate-x-0"
        )}
      >
        <div className="p-6 border-bottom border-slate-800">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Cpu className="text-white" size={24} />
            </div>
            <h1 className="text-xl font-bold text-white tracking-tight">Traffic SegNet</h1>
          </div>
          <p className="text-xs text-slate-400 font-mono uppercase tracking-[0.2em]">Research Project v12.0</p>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  if (window.innerWidth < 1024) setIsOpen(false);
                }}
                className={cn(
                  "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group text-left",
                  isActive 
                    ? "bg-blue-600/10 text-blue-400 border border-blue-600/20" 
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                )}
              >
                <Icon size={20} className={cn(isActive ? "text-blue-400" : "text-slate-500 group-hover:text-slate-200")} />
                <span className="font-medium">{item.label}</span>
                {isActive && (
                  <motion.div 
                    layoutId="active-indicator"
                    className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400"
                  />
                )}
              </button>
            );
          })}
        </nav>

        <div className="p-6 mt-auto border-t border-slate-900 bg-slate-950/50">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center">
              <Database size={14} className="text-slate-400" />
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-300">Dataset Alpha</p>
              <p className="text-[10px] text-slate-500 font-mono">5,420 IMAGES</p>
            </div>
          </div>
          <div className="text-[10px] text-slate-600 mt-4 leading-relaxed font-mono">
            © 2026 RESEARCH TEAM<br />
            MULTI-CLASS TRAFFIC VIOLATION
          </div>
        </div>
      </motion.aside>

      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/60 z-30" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
};
