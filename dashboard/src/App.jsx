
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, ComposedChart, YAxis as ReYAxis
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

// Optimization: Pre-define styles to avoid re-calculation
const CHART_MARGIN = { top: 5, right: 5, left: 5, bottom: 5 };

const Dashboard = () => {
  const [data, setData] = useState(() => {
    const saved = localStorage.getItem('axon_chart_data');
    return saved ? JSON.parse(saved) : [];
  });
  const [trades, setTrades] = useState(() => {
    const saved = localStorage.getItem('axon_trade_history');
    return saved ? JSON.parse(saved) : [];
  });

  const [stats, setStats] = useState({
    timestep: 0, equity: 10000, reward: 0, 
    value_loss: 0, policy_loss: 0, entropy: 1.0, 
    lr: 0, fps: 0, regime: 0, status: 'STANDBY',
    pos_size: 0, pos_pnl: 0, last_price: 0, 
    symbol: 'BTCUSDm', task: 'INITIALIZING', total_steps: 1000000
  });

  const lastTimestepRef = useRef(-1);

  useEffect(() => {
    const fetchPulse = async () => {
      try {
        const response = await fetch('/telemetry.json?nocache=' + Date.now());
        if (!response.ok) return;
        const msg = await response.json();
        
        if (msg && msg.timestep !== lastTimestepRef.current) {
          lastTimestepRef.current = msg.timestep;
          
          // Batch updates to reduce re-renders
          setStats(prev => ({ ...prev, ...msg, status: 'ACTIVE' }));
          
          setData(prev => {
            const newData = [...prev, { ...msg, time: msg.timestep }].slice(-150); // Slightly smaller window for speed
            localStorage.setItem('axon_chart_data', JSON.stringify(newData));
            return newData;
          });

          if (msg.trades && Array.isArray(msg.trades) && msg.trades.length > 0) {
            setTrades(prev => {
              const filteredNew = msg.trades.filter(nt => !prev.some(pt => pt.id === nt.id));
              if (filteredNew.length === 0) return prev;
              const updated = [...filteredNew, ...prev].slice(0, 50); // Smaller blotter
              localStorage.setItem('axon_trade_history', JSON.stringify(updated));
              return updated;
            });
          }
        }
      } catch (e) {}
    };

    // Use a slightly slower fetch (300ms) to give the CPU room to breathe for AI
    const timer = setInterval(fetchPulse, 300);
    return () => clearInterval(timer);
  }, []);

  const progress = (stats.timestep / stats.total_steps) * 100;
  const isOptimizing = stats.task === 'OPTIMIZING';
  
  // Memoize charts to prevent unnecessary re-renders when only stats change
  const MainChart = useMemo(() => (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={CHART_MARGIN}>
        <CartesianGrid stroke="#111" vertical={false} strokeDasharray="3 3" />
        <XAxis dataKey="time" hide />
        <ReYAxis yAxisId="price" domain={['auto', 'auto']} hide />
        <ReYAxis yAxisId="equity" orientation="right" domain={['auto', 'auto']} hide />
        <Line yAxisId="price" type="monotone" dataKey="last_price" stroke="#444" strokeWidth={1} dot={false} isAnimationActive={false} />
        <Line yAxisId="equity" type="monotone" dataKey="equity" stroke="#00ff88" strokeWidth={2} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  ), [data]);

  const RewardChart = useMemo(() => (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={CHART_MARGIN}>
        <Area type="step" dataKey="reward" stroke="#fff" fill="#fff" fillOpacity={0.05} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  ), [data]);

  return (
    <div style={{ 
      background: '#000', color: '#fff', height: '100vh', width: '100vw',
      display: 'grid', gridTemplateColumns: '300px 1fr', gridTemplateRows: '80px 1fr',
      overflow: 'hidden', fontFamily: 'monospace'
    }}>
      
      <AnimatePresence>
        {isOptimizing && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            style={{ 
              position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.9)', zIndex: 1000, 
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              backdropFilter: 'blur(8px)'
            }}
          >
            <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.5 }}
              style={{ fontSize: '0.8rem', letterSpacing: '10px', color: '#00ff88' }}>
              NEURAL_UPDATE_IN_PROGRESS
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <header style={{ 
        gridColumn: '1 / 3', borderBottom: '1px solid #111', 
        display: 'flex', alignItems: 'center', padding: '0 30px', gap: '30px'
      }}>
        <div style={{ fontSize: '0.9rem', fontWeight: 900, letterSpacing: '2px' }}>AXON_APEX</div>
        
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.5rem', marginBottom: '4px', opacity: 0.5 }}>
            <span>{stats.task}</span>
            <span>{progress.toFixed(2)}%</span>
          </div>
          <div style={{ height: '1px', background: '#222' }}>
            <div style={{ height: '100%', background: '#fff', width: `${progress}%`, transition: 'width 0.3s ease' }} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '30px', fontSize: '0.7rem' }}>
          <Stat label="SYNC" value={`${(stats.fps || 0).toFixed(0)}Hz`} />
          <Stat label="BTC" value={stats.last_price > 0 ? `$${stats.last_price.toFixed(0)}` : '...'} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
             <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: stats.status === 'ACTIVE' ? '#00ff88' : '#ff3b30' }} />
             <span style={{ fontWeight: 800, fontSize: '0.6rem' }}>{stats.status}</span>
          </div>
        </div>
      </header>

      <aside style={{ borderRight: '1px solid #111', padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px', overflow: 'hidden' }}>
        <FinanceBlock label="EQUITY" value={`$${(stats.equity || 10000).toFixed(2)}`} />
        
        <div style={{ border: '1px solid #111', padding: '15px', background: '#050505' }}>
          <div style={{ fontSize: '0.5rem', opacity: 0.4, marginBottom: '5px' }}>EXPOSURE</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 900 }}>{stats.pos_size === 0 ? 'NEUTRAL' : (stats.pos_size > 0 ? 'LONG' : 'SHORT')}</span>
            <span style={{ fontSize: '0.8rem', color: stats.pos_pnl >= 0 ? '#00ff88' : '#ff3b30' }}>{stats.pos_pnl.toFixed(2)}</span>
          </div>
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ fontSize: '0.5rem', opacity: 0.4, marginBottom: '10px' }}>RECENT_TRADES</div>
          <div style={{ flex: 1, overflowY: 'auto', fontSize: '0.55rem' }}>
            {trades.map((t, i) => (
              <div key={t.id || i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #080808' }}>
                <span style={{ color: t.type.includes('BUY') ? '#00ff88' : '#ff3b30' }}>{t.type}</span>
                <span style={{ opacity: 0.6 }}>{Math.abs(t.size).toFixed(4)}</span>
                <span style={{ color: t.pnl >= 0 ? '#00ff88' : '#ff3b30' }}>{t.pnl.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <main style={{ display: 'grid', gridTemplateRows: '1fr 200px', overflow: 'hidden' }}>
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: '0.5rem', opacity: 0.3, marginBottom: '10px' }}>CONVERGENCE_STREAM</div>
          <div style={{ flex: 1 }}>{MainChart}</div>
        </div>

        <div style={{ borderTop: '1px solid #111', display: 'grid', gridTemplateColumns: '1fr 250px' }}>
          <div style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: '0.5rem', opacity: 0.3, marginBottom: '10px' }}>REWARD_GRADIENT</div>
            <div style={{ flex: 1 }}>{RewardChart}</div>
          </div>
          <div style={{ borderLeft: '1px solid #111', padding: '20px', background: '#030303' }}>
            <div style={{ fontSize: '0.5rem', opacity: 0.3, marginBottom: '15px' }}>KERNEL_STATUS</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <Row label="REGIME" value={`0x${stats.regime}`} />
              <Row label="V_LOSS" value={stats.value_loss.toFixed(4)} />
              <Row label="ENTROPY" value={stats.entropy.toFixed(3)} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

const Stat = ({ label, value }) => (
  <div style={{ textAlign: 'right' }}>
    <div style={{ fontSize: '0.5rem', opacity: 0.4 }}>{label}</div>
    <div style={{ fontWeight: 900 }}>{value}</div>
  </div>
);

const FinanceBlock = ({ label, value }) => (
  <div>
    <div style={{ fontSize: '0.5rem', opacity: 0.4, marginBottom: '4px' }}>{label}</div>
    <div style={{ fontSize: '1.2rem', fontWeight: 900 }}>{value}</div>
  </div>
);

const Row = ({ label, value }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem' }}>
    <span style={{ opacity: 0.4 }}>{label}</span>
    <span style={{ fontWeight: 800 }}>{value}</span>
  </div>
);

export default Dashboard;
