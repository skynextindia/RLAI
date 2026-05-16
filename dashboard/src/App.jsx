
import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const POLL_MS = 500;
const MAX_POINTS = 200;

export default function Dashboard() {
  const [stats, setStats] = useState({
    step: 0, total_steps: 1000000, equity: 10000, pnl: 0,
    entropy: 1.0, value_loss: 0, fps: 0,
    sharpe_100: 0, win_rate: 0, last_price: 0,
    pos_size: 0, pos_pnl: 0, task: 'CONNECTING', status: 'STANDBY',
    convergence_stream: [], reward_gradient: [],
    audit: {}
  });
  const [equityHistory, setEquityHistory] = useState([]);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [isStale, setIsStale] = useState(false);
  const lastStepRef = useRef(-1);
  const lastUpdateRef = useRef(Date.now());

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8080/telemetry', { cache: 'no-store' });
        if (!res.ok) return;
        const msg = await res.json();

        // KEY FIX: backend sends 'step', not 'timestep'
        const step = msg.step ?? msg.timestep ?? 0;

        if (step !== lastStepRef.current) {
          lastStepRef.current = step;
          lastUpdateRef.current = Date.now();
          setIsStale(false);
          setStats(prev => ({ ...prev, ...msg, step }));

          setEquityHistory(prev => {
            const next = [...prev, { step, equity: msg.equity ?? 10000, pnl: msg.pnl ?? 0 }];
            return next.slice(-MAX_POINTS);
          });

          const rg = msg.reward_gradient ?? [];
          if (rg.length > 0) {
            setRewardHistory(rg.map((v, i) => ({ i, r: v })));
          }
        } else {
          if (Date.now() - lastUpdateRef.current > 8000) setIsStale(true);
        }
      } catch (_) {
        if (Date.now() - lastUpdateRef.current > 8000) setIsStale(true);
      }
    };

    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const progress = Math.min(100, ((stats.step ?? 0) / (stats.total_steps || 1)) * 100);
  const isActive = !isStale && stats.step > 0;
  const pulseColor = isActive ? '#00ff88' : '#ff3b30';

  // Neural stream from convergence_stream array
  const neuralStream = (stats.convergence_stream ?? []).map((v, i) => ({ i, v }));

  const audit = stats.audit ?? {};
  const winRate = ((stats.win_rate ?? 0) * 100).toFixed(0);
  const sharpe = (stats.sharpe_100 ?? 0).toFixed(3);
  const expectancy = audit.expectancy != null ? audit.expectancy.toFixed(4) : '—';
  const profitFactor = audit.profit_factor != null ? audit.profit_factor.toFixed(2) : '—';

  return (
    <div style={{
      background: '#020304', color: '#fff', height: '100vh', width: '100vw',
      display: 'grid', gridTemplateColumns: '260px 1fr', gridTemplateRows: '60px 1fr 180px',
      overflow: 'hidden', fontFamily: '"SF Mono", monospace', fontSize: '11px'
    }}>

      {/* HEADER */}
      <header style={{
        gridColumn: '1/3', borderBottom: '1px solid #0f1a12',
        display: 'flex', alignItems: 'center', padding: '0 24px', gap: '32px',
        background: 'linear-gradient(90deg, #020304 0%, #03080a 100%)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <motion.div
            animate={{ scale: isActive ? [1, 1.3, 1] : 1, opacity: isActive ? [1, 0.6, 1] : 0.3 }}
            transition={{ repeat: Infinity, duration: 1.2 }}
            style={{ width: 7, height: 7, borderRadius: '50%', background: pulseColor, boxShadow: isActive ? `0 0 8px ${pulseColor}` : 'none' }}
          />
          <span style={{ fontWeight: 900, letterSpacing: '3px', fontSize: '13px' }}>AXON_APEX</span>
        </div>

        {/* Progress bar */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.45, fontSize: '9px', marginBottom: 4 }}>
            <span>MISSION: {stats.task}</span>
            <span>STEP {(stats.step ?? 0).toLocaleString()} / {(stats.total_steps ?? 0).toLocaleString()} — {progress.toFixed(2)}%</span>
          </div>
          <div style={{ height: 2, background: '#0f1a12', borderRadius: 1 }}>
            <motion.div animate={{ width: `${progress}%` }} transition={{ type: 'spring', stiffness: 60 }}
              style={{ height: '100%', background: 'linear-gradient(90deg, #00ff88, #00ccff)', borderRadius: 1 }} />
          </div>
        </div>

        <HeaderStat label="PULSE" value={`${(stats.fps ?? 0).toFixed(0)}Hz`} />
        <HeaderStat label="SHARPE" value={sharpe} color={parseFloat(sharpe) > 0 ? '#00ff88' : '#ff4444'} />
        <HeaderStat label="W_RATE" value={`${winRate}%`} color={parseInt(winRate) > 50 ? '#00ff88' : '#aaa'} />
        <HeaderStat label="EXP/T" value={expectancy} color={parseFloat(expectancy) > 0 ? '#00ff88' : '#ff4444'} />
        <HeaderStat label="BTC" value={stats.last_price > 0 ? `$${Math.floor(stats.last_price).toLocaleString()}` : '...'} color="#00ccff" />

        {isStale && (
          <div style={{ background: '#ff3b30', color: '#fff', padding: '4px 12px', fontSize: '9px', fontWeight: 900, letterSpacing: 2 }}>
            SIGNAL_LOST
          </div>
        )}
      </header>

      {/* SIDEBAR */}
      <aside style={{ borderRight: '1px solid #0f1a12', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', gridRow: '2/4', overflowY: 'auto' }}>
        <Block label="EQUITY">
          <div style={{ fontSize: '22px', fontWeight: 900, color: '#00ff88', textShadow: '0 0 20px rgba(0,255,136,0.3)' }}>
            ${(stats.equity ?? 10000).toFixed(2)}
          </div>
          <div style={{ fontSize: '10px', opacity: 0.5, marginTop: 2 }}>
            {stats.pnl >= 0 ? '+' : ''}{(stats.pnl ?? 0).toFixed(2)} PnL
          </div>
        </Block>

        <Block label="EXPOSURE">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 900, fontSize: '14px', color: stats.pos_size > 0 ? '#00ff88' : stats.pos_size < 0 ? '#ff4444' : '#666' }}>
              {stats.pos_size === 0 ? 'NEUTRAL' : stats.pos_size > 0 ? 'LONG' : 'SHORT'}
            </span>
            <span style={{ color: stats.pos_pnl >= 0 ? '#00ff88' : '#ff4444' }}>
              {(stats.pos_pnl ?? 0).toFixed(2)}
            </span>
          </div>
          <div style={{ opacity: 0.4, marginTop: 4 }}>{Math.abs(stats.pos_size ?? 0).toFixed(4)} BTC</div>
        </Block>

        <Block label="KERNEL_STATUS">
          <KV label="V_LOSS" value={(stats.value_loss ?? 0).toFixed(4)} />
          <KV label="ENTROPY" value={(stats.entropy ?? 0).toFixed(3)} />
          <KV label="PF" value={profitFactor} color={parseFloat(profitFactor) > 1 ? '#00ff88' : '#ff4444'} />
          <KV label="EXP/T" value={expectancy} color={parseFloat(expectancy) > 0 ? '#00ff88' : '#ff4444'} />
        </Block>

        <Block label="REWARD_DECOMP" style={{ flex: 1 }}>
          {Object.entries(audit.decomposition ?? {}).map(([k, v]) => (
            <KV key={k} label={k.toUpperCase()} value={(v ?? 0).toFixed(3)} color={v >= 0 ? '#00ff88' : '#ff4444'} />
          ))}
          {Object.keys(audit.decomposition ?? {}).length === 0 && (
            <div style={{ opacity: 0.3, fontSize: '10px' }}>awaiting trades...</div>
          )}
        </Block>
      </aside>

      {/* MAIN CHART — Equity Curve */}
      <div style={{ padding: '16px 16px 8px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ fontSize: '9px', opacity: 0.3, letterSpacing: 2, marginBottom: 8 }}>EQUITY_CURVE ({equityHistory.length} pts)</div>
        {equityHistory.length > 1 ? (
          <div style={{ flex: 1 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityHistory} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00ff88" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#0a0a0a" vertical={false} />
                <XAxis dataKey="step" hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Tooltip
                  contentStyle={{ background: '#060a07', border: '1px solid #0f1a12', fontSize: '10px' }}
                  formatter={(v) => [`$${v.toFixed(2)}`, 'Equity']}
                />
                <Area type="monotone" dataKey="equity" stroke="#00ff88" fill="url(#eq)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.2 }}>
            <motion.div animate={{ opacity: [0.2, 0.6, 0.2] }} transition={{ repeat: Infinity, duration: 2 }}>
              AWAITING_SIGNAL...
            </motion.div>
          </div>
        )}
      </div>

      {/* BOTTOM ROW — Reward + Neural Stream */}
      <div style={{ borderTop: '1px solid #0f1a12', display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
        {/* Reward Gradient */}
        <div style={{ padding: '12px 16px', borderRight: '1px solid #0f1a12', display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: '9px', opacity: 0.3, letterSpacing: 2, marginBottom: 8 }}>REWARD_GRADIENT</div>
          {rewardHistory.length > 1 ? (
            <div style={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rewardHistory} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                  <XAxis dataKey="i" hide />
                  <YAxis domain={['auto', 'auto']} hide />
                  <Line type="basis" dataKey="r" stroke="#00ccff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.15, fontSize: '10px' }}>
              NO_REWARD_HISTORY
            </div>
          )}
        </div>

        {/* Neural Stream */}
        <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: '9px', opacity: 0.3, letterSpacing: 2, marginBottom: 8 }}>NEURAL_STREAM ({neuralStream.length} dims)</div>
          {neuralStream.length > 1 ? (
            <div style={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={neuralStream} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                  <XAxis dataKey="i" hide />
                  <YAxis domain={['auto', 'auto']} hide />
                  <Line type="basis" dataKey="v" stroke="#7c5cbf" strokeWidth={1} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.15, fontSize: '10px' }}>
              NO_NEURAL_STREAM
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const Block = ({ label, children, style = {} }) => (
  <div style={{ border: '1px solid #0f1a12', padding: '12px', background: '#030607', ...style }}>
    <div style={{ fontSize: '9px', opacity: 0.35, letterSpacing: 2, marginBottom: 8 }}>{label}</div>
    {children}
  </div>
);

const KV = ({ label, value, color = '#fff' }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid #060a07' }}>
    <span style={{ opacity: 0.45 }}>{label}</span>
    <span style={{ fontWeight: 700, color }}>{value}</span>
  </div>
);

const HeaderStat = ({ label, value, color = '#fff' }) => (
  <div style={{ textAlign: 'right' }}>
    <div style={{ fontSize: '8px', opacity: 0.4, letterSpacing: 1, marginBottom: 2 }}>{label}</div>
    <div style={{ fontWeight: 900, color, fontSize: '12px' }}>{value}</div>
  </div>
);
