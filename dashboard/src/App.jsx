
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
    sharpe_100: 0, win_rate: 0,
    symbol: 'BTCUSDm', task: 'INITIALIZING', total_steps: 1000000
  });

  const lastTimestepRef = useRef(-1);
  const lastUpdateRef = useRef(Date.now());
  const [isStale, setIsStale] = useState(false);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const host = window.location.hostname || '127.0.0.1';
        const response = await fetch(`http://${host}:8080/telemetry`);
        if (!response.ok) return;
        const msg = await response.json();
        
        if (msg && msg.timestep !== lastTimestepRef.current) {
          lastTimestepRef.current = msg.timestep;
          lastUpdateRef.current = Date.now();
          setIsStale(false);
          
          setStats(prev => ({ ...prev, ...msg, status: msg.task === 'COMPLETE' ? 'FINISHED' : 'ACTIVE' }));
          
          setData(prev => {
            const newData = [...prev, { ...msg, time: msg.timestep }].slice(-150);
            localStorage.setItem('axon_chart_data', JSON.stringify(newData));
            return newData;
          });

          if (msg.trades && Array.isArray(msg.trades) && msg.trades.length > 0) {
            setTrades(prev => {
              const filteredNew = msg.trades.filter(nt => !prev.some(pt => pt.id === nt.id));
              if (filteredNew.length === 0) return prev;
              const updated = [...filteredNew, ...prev].slice(0, 50);
              localStorage.setItem('axon_trade_history', JSON.stringify(updated));
              return updated;
            });
          }
        } else {
          // If no update for 5 seconds, mark as stale (unless complete)
          if (Date.now() - lastUpdateRef.current > 5000 && stats.task !== 'COMPLETE') {
            setIsStale(true);
          }
        }
      } catch (e) {}
    };

    const timer = setInterval(fetchTelemetry, 300);
    return () => clearInterval(timer);
  }, [stats.task]);

  const progress = (stats.timestep / stats.total_steps) * 100;
  const isOptimizing = stats.task === 'OPTIMIZING';
  const isComplete = stats.task === 'COMPLETE' || stats.status === 'MISSION_ACCOMPLISHED';
  
  // Memoize charts... (rest of the logic remains same)
  
  // Memoize charts to prevent unnecessary re-renders when only stats change
  const MainChart = useMemo(() => (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={CHART_MARGIN}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3}/>
            <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#111" vertical={false} strokeDasharray="3 3" />
        <XAxis dataKey="time" hide />
        <ReYAxis yAxisId="price" domain={['auto', 'auto']} hide />
        <ReYAxis yAxisId="equity" orientation="right" domain={['auto', 'auto']} hide />
        <Line yAxisId="price" type="basis" dataKey="last_price" stroke="#333" strokeWidth={1} dot={false} isAnimationActive={false} />
        <Area yAxisId="equity" type="monotone" dataKey="equity" stroke="#00ff88" fill="url(#equityGradient)" strokeWidth={3} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  ), [data]);

  const RewardChart = useMemo(() => (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={CHART_MARGIN}>
        <Area type="stepAfter" dataKey="reward" stroke="#fff" fill="#fff" fillOpacity={0.1} isAnimationActive={false} />
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

        {isComplete && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            style={{ 
              position: 'absolute', inset: 0, background: 'rgba(0,10,5,0.95)', zIndex: 2000, 
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              backdropFilter: 'blur(12px)', border: '2px solid #00ff88'
            }}
          >
            <div style={{ fontSize: '1.5rem', fontWeight: 900, color: '#00ff88', marginBottom: '10px' }}>MISSION_ACCOMPLISHED</div>
            <div style={{ fontSize: '0.7rem', opacity: 0.7, letterSpacing: '2px', marginBottom: '40px' }}>NEURAL_WEIGHTS_CALIBRATED_AND_SAVED</div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', width: '300px', fontSize: '0.6rem', borderTop: '1px solid #111', paddingTop: '20px' }}>
              <div style={{ opacity: 0.5 }}>FINAL_EQUITY</div>
              <div style={{ textAlign: 'right', fontWeight: 800 }}>${stats.equity.toFixed(2)}</div>
              <div style={{ opacity: 0.5 }}>TOTAL_STEPS</div>
              <div style={{ textAlign: 'right', fontWeight: 800 }}>{stats.total_steps.toLocaleString()}</div>
              <div style={{ opacity: 0.5 }}>TARGET_SYMBOL</div>
              <div style={{ textAlign: 'right', fontWeight: 800 }}>{stats.symbol}</div>
            </div>

            <div style={{ marginTop: '50px', fontSize: '0.6rem', padding: '10px 40px', background: '#00ff88', color: '#000', fontWeight: 900, cursor: 'pointer', letterSpacing: '2px' }} onClick={() => window.location.reload()}>RE-INITIALIZE</div>
          </motion.div>
        )}

        {isStale && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            style={{ 
              position: 'absolute', top: '90px', right: '30px', background: '#ff3b30', color: '#fff', 
              padding: '10px 20px', fontSize: '0.6rem', fontWeight: 900, zIndex: 500
            }}
          >
            SIGNAL_LOST: ENGINE_HALTED
          </motion.div>
        )}
      </AnimatePresence>

      <header style={{ 
        gridColumn: '1 / 3', borderBottom: '1px solid #111', 
        display: 'flex', alignItems: 'center', padding: '0 30px', gap: '40px',
        background: 'linear-gradient(90deg, #000 0%, #050505 100%)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <motion.div 
            animate={{ scale: stats.status === 'ACTIVE' ? [1, 1.2, 1] : 1 }}
            transition={{ repeat: Infinity, duration: 1.5 }}
            style={{ width: '8px', height: '8px', borderRadius: '50%', background: stats.status === 'ACTIVE' ? '#00ff88' : '#ff3b30', boxShadow: stats.status === 'ACTIVE' ? '0 0 10px #00ff88' : 'none' }} 
          />
          <div style={{ fontSize: '0.9rem', fontWeight: 900, letterSpacing: '3px' }}>AXON_APEX</div>
        </div>
        
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.5rem', marginBottom: '6px', opacity: 0.6, letterSpacing: '1px' }}>
            <span>MISSION: {stats.task}</span>
            <span>{progress.toFixed(2)}%</span>
          </div>
          <div style={{ height: '2px', background: '#111', borderRadius: '1px', overflow: 'hidden' }}>
            <motion.div 
              animate={{ width: `${progress}%` }}
              style={{ height: '100%', background: 'linear-gradient(90deg, #00ff88, #fff)' }} 
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '40px', fontSize: '0.7rem' }}>
          <Stat label="PULSE" value={`${(stats.fps || 0).toFixed(0)}Hz`} />
          <Stat label="SHARPE" value={(stats.sharpe_100 || 0).toFixed(2)} />
          <Stat label="W_RATE" value={`${((stats.win_rate || 0) * 100).toFixed(0)}%`} />
          <Stat label="BTC_INDEX" value={stats.last_price > 0 ? `$${stats.last_price.toLocaleString()}` : '...'} color="#00ff88" />
          <div style={{ cursor: 'pointer', opacity: 0.3, alignSelf: 'center' }} onClick={() => { localStorage.clear(); window.location.reload(); }}>RESET</div>
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

const Stat = ({ label, value, color = '#fff' }) => (
  <div style={{ textAlign: 'right' }}>
    <div style={{ fontSize: '0.45rem', opacity: 0.5, letterSpacing: '1px', marginBottom: '2px' }}>{label}</div>
    <div style={{ fontWeight: 900, color, fontSize: '0.8rem' }}>{value}</div>
  </div>
);

const FinanceBlock = ({ label, value }) => (
  <div style={{ background: '#050505', padding: '15px', border: '1px solid #111' }}>
    <div style={{ fontSize: '0.5rem', opacity: 0.5, marginBottom: '6px', letterSpacing: '1px' }}>{label}</div>
    <div style={{ fontSize: '1.4rem', fontWeight: 900, color: '#00ff88', textShadow: '0 0 20px rgba(0,255,136,0.2)' }}>{value}</div>
  </div>
);

const Row = ({ label, value }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem' }}>
    <span style={{ opacity: 0.4 }}>{label}</span>
    <span style={{ fontWeight: 800 }}>{value}</span>
  </div>
);

export default Dashboard;
