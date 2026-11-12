import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import StatusBanner from './StatusBanner';
import Sidebar from './Sidebar';
import DevMockPanel from '../../mocks/DevMockPanel';

export function Shell({ children }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <StatusBanner />
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-8 bg-slate-950">
          <div className="max-w-7xl mx-auto w-full">
            {children || <Outlet />}
          </div>
        </main>
      </div>
      <DevMockPanel />
    </div>
  );
}

export default Shell;
