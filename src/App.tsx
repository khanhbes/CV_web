/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Introduction } from './components/views/Introduction';
import { Quantitative } from './components/views/Quantitative';
import { Qualitative } from './components/views/Qualitative';
import { LiveInference } from './components/views/LiveInference';
import { motion, AnimatePresence } from 'motion/react';

export default function App() {
  const [activeTab, setActiveTab] = useState('introduction');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const renderContent = () => {
    switch (activeTab) {
      case 'introduction': return <Introduction />;
      case 'quantitative': return <Quantitative />;
      case 'qualitative': return <Qualitative />;
      case 'inference': return <LiveInference />;
      default: return <Introduction />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex selection:bg-blue-500/30 selection:text-white">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
      />
      
      <main className="flex-1 lg:ml-[280px] min-h-screen relative">
        {/* Background Gradients */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
          <div className="absolute top-[-10%] left-[20%] w-[50%] h-[50%] bg-blue-600/5 blur-[120px] rounded-full" />
          <div className="absolute bottom-[-10%] right-[10%] w-[40%] h-[40%] bg-purple-600/5 blur-[120px] rounded-full" />
        </div>

        <div className="relative z-10 p-6 lg:p-12">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2, ease: "easeInOut" }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
