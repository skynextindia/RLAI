
import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const POLL_MS = 500;
const MAX_POINTS = 2000;

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

          if (msg.equity_history && msg.equity_history.length > 0) {
            setEquityHistory(msg.equity_history.slice(-MAX_POINTS));
          } else {
            setEquityHistory(prev => {
              const next = [...prev, { step, equity: msg.equity ?? 10000, pnl: msg.pnl ?? 0 }];
              return next.slice(-MAX_POINTS);
            });
          }

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

  // Calculate Remaining Time (ETA)
  const remainingSteps = Math.max(0, (stats.total_steps ?? 0) - (stats.step ?? 0));
  const fps = stats.fps ?? 0;
  let remainingTimeStr = '—';
  if (fps > 0 && remainingSteps > 0) {
    const totalSeconds = remainingSteps / fps;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);
    if (hours > 0) {
      remainingTimeStr = `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      remainingTimeStr = `${minutes}m ${seconds}s`;
    } else {
      remainingTimeStr = `${seconds}s`;
    }
  }

  // Reward Gradient Metrics
  const rewardGradValues = stats.reward_gradient ?? [];
  const latestReward = rewardGradValues.length > 0 ? rewardGradValues[rewardGradValues.length - 1] : 0.0;
  const meanReward = rewardGradValues.length > 0 ? rewardGradValues.reduce((a, b) => a + b, 0) / rewardGradValues.length : 0.0;
  const maxReward = rewardGradValues.length > 0 ? Math.max(...rewardGradValues) : 0.0;
  const minReward = rewardGradValues.length > 0 ? Math.min(...rewardGradValues) : 0.0;

  // Neural Stream Metrics
  const neuralValues = stats.convergence_stream ?? [];
  const meanNeural = neuralValues.length > 0 ? neuralValues.reduce((a, b) => a + b, 0) / neuralValues.length : 0.0;
  const maxNeural = neuralValues.length > 0 ? Math.max(...neuralValues) : 0.0;
  const minNeural = neuralValues.length > 0 ? Math.min(...neuralValues) : 0.0;

  return (
    <div style={{
      background: '#020304', color: '#fff', height: '100vh', width: '100vw',
      display: 'grid', gridTemplateColumns: '260px 1fr', gridTemplateRows: '60px 1fr',
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
        <div style={{ flex: 1, padding: '0 8px' }}>
          {(() => {
            const stepVal = stats.step ?? 0;
            const windowIdx = Math.floor(stepVal / 10000) + 1;
            const windowStartStep = (windowIdx - 1) * 10000;
            const windowEndStep = windowIdx * 10000;
            
            let currentStage = 'STAGE 1: SURVIVAL & DISCOVERY';
            let stageColor = '#888888';
            if (windowIdx > 80) {
              currentStage = 'STAGE 4: INSTITUTIONAL EXPLOITATION';
              stageColor = '#00ff88';
            } else if (windowIdx > 50) {
              currentStage = 'STAGE 3: REGIME CONDITIONING';
              stageColor = '#00ccff';
            } else if (windowIdx > 20) {
              currentStage = 'STAGE 2: TREND ALIGNMENT';
              stageColor = '#ffbb00';
            }
            
            return (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', marginBottom: 4, gap: '12px' }}>
                  <span style={{ opacity: 0.45 }}>MISSION: {stats.task}</span>
                  <span style={{ color: stageColor, fontWeight: 900, textShadow: `0 0 10px ${stageColor}33` }}>{currentStage}</span>
                  <span style={{ opacity: 0.8, color: '#00ccff', fontWeight: 'bold' }}>W_{windowIdx} ({windowStartStep.toLocaleString()} - {windowEndStep.toLocaleString()})</span>
                  <span style={{ opacity: 0.45 }}>STEP {stepVal.toLocaleString()} / {(stats.total_steps ?? 0).toLocaleString()} — {progress.toFixed(2)}% (ETA: {remainingTimeStr})</span>
                </div>
                <div style={{ height: 2, background: '#0f1a12', borderRadius: 1 }}>
                  <motion.div animate={{ width: `${progress}%` }} transition={{ type: 'spring', stiffness: 60 }}
                    style={{ height: '100%', background: `linear-gradient(90deg, #00ff88, ${stageColor})`, borderRadius: 1 }} />
                </div>
              </>
            );
          })()}
        </div>

        <HeaderStat label="PULSE" value={`${(stats.fps ?? 0).toFixed(0)}Hz`} />
        <HeaderStat label="SHARPE" value={sharpe} color={parseFloat(sharpe) > 0 ? '#00ff88' : '#ff4444'} />
        <HeaderStat label="W_RATE" value={`${winRate}%`} color={parseInt(winRate) > 50 ? '#00ff88' : '#aaa'} />
        <HeaderStat label="EXP/T" value={expectancy} color={parseFloat(expectancy) > 0 ? '#00ff88' : '#ff4444'} />
        <HeaderStat label="EUR/USD" value={stats.last_price > 0 ? stats.last_price.toFixed(4) : '...'} color="#00ccff" />

        {isStale && (
          <div style={{ background: '#ff3b30', color: '#fff', padding: '4px 12px', fontSize: '9px', fontWeight: 900, letterSpacing: 2 }}>
            SIGNAL_LOST
          </div>
        )}
      </header>

      {/* SIDEBAR */}
      <aside style={{ borderRight: '1px solid #0f1a12', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', gridRow: '2/3', overflowY: 'auto' }}>
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
          <div style={{ opacity: 0.4, marginTop: 4 }}>{Math.abs(stats.pos_size ?? 0).toFixed(4)} LOTS</div>
        </Block>

        <Block label="KERNEL_STATUS">
          <KV label="V_LOSS" value={(stats.value_loss ?? 0).toFixed(4)} />
          <KV label="ENTROPY" value={(stats.entropy ?? 0).toFixed(3)} />
          <KV label="PF" value={profitFactor} color={parseFloat(profitFactor) > 1 ? '#00ff88' : '#ff4444'} />
          <KV label="EXP/T" value={expectancy} color={parseFloat(expectancy) > 0 ? '#00ff88' : '#ff4444'} />
        </Block>

        <Block label="REWARD_DECOMP">
          {Object.entries(audit.decomposition ?? {}).map(([k, v]) => (
            <KV key={k} label={k.toUpperCase()} value={(v ?? 0).toFixed(3)} color={v >= 0 ? '#00ff88' : '#ff4444'} />
          ))}
          {Object.keys(audit.decomposition ?? {}).length === 0 && (
            <div style={{ opacity: 0.3, fontSize: '10px' }}>awaiting trades...</div>
          )}
        </Block>

        <Block label="REWARD_GRADIENT">
          <KV label="LATEST" value={latestReward.toFixed(4)} color={latestReward >= 0 ? '#00ff88' : '#ff4444'} />
          <KV label="MEAN" value={meanReward.toFixed(4)} color={meanReward >= 0 ? '#00ff88' : '#ff4444'} />
          <KV label="MAX" value={maxReward.toFixed(4)} color="#00ff88" />
          <KV label="MIN" value={minReward.toFixed(4)} color="#ff4444" />
        </Block>

        <Block label="NEURAL_STREAM">
          <KV label="DIMS" value={neuralValues.length.toString()} color="#fff" />
          <KV label="MEAN_ACT" value={meanNeural.toFixed(6)} color={meanNeural >= 0 ? '#00ccff' : '#7c5cbf'} />
          <KV label="PEAK_ACT" value={maxNeural.toFixed(6)} color="#00ccff" />
          <KV label="FLOOR_ACT" value={minNeural.toFixed(6)} color="#7c5cbf" />
          <KV label="ACTIVE" value={neuralValues.filter(v => Math.abs(v) > 1e-5).length.toString()} color="#fff" />
        </Block>
      </aside>

      {/* MAIN BODY — Chart + Recent Trades */}
      <main style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', height: '100%', overflow: 'hidden' }}>
        {/* MAIN CHART — Equity Curve */}
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #0f1a12', height: '100%' }}>
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
                  <YAxis 
                    domain={[dataMin => Math.min(9900, dataMin - 10), dataMax => Math.max(10100, dataMax + 10)]} 
                    tick={{ fill: 'rgba(0, 255, 136, 0.4)', fontSize: 9 }}
                    axisLine={false}
                    tickLine={false}
                    width={45}
                  />
                   <Tooltip
                    contentStyle={{ background: '#060a07', border: '1px solid #0f1a12', fontSize: '10px', borderRadius: '4px' }}
                    formatter={(v, name) => {
                      if (name === 'equity') return [`$${v.toFixed(2)}`, 'Equity'];
                      return [v, name];
                    }}
                    labelFormatter={(label, items) => {
                      const payload = items[0]?.payload;
                      if (payload) {
                        return (
                          <div style={{ fontFamily: 'monospace' }}>
                            <div style={{ color: '#aaa', marginBottom: 2 }}>Step: {label}</div>
                            <div style={{ color: '#00ff88', fontWeight: 'bold' }}>Win Rate: {payload.win_rate ? `${payload.win_rate.toFixed(1)}%` : 'N/A'}</div>
                          </div>
                        );
                      }
                      return `Step: ${label}`;
                    }}
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

        {/* Recent Trades Table */}
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', overflowY: 'auto', height: '100%' }}>
          <div style={{ fontSize: '9px', opacity: 0.3, letterSpacing: 2, marginBottom: 8 }}>RECENT_TRADES</div>
          {(stats.recent_trades ?? []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '8px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #0f1a12', opacity: 0.5, color: '#aaa' }}>
                  <th style={{ padding: '4px 0' }}>TIME</th>
                  <th>STEP</th>
                  <th>SIDE</th>
                  <th>ENTRY</th>
                  <th>EXIT</th>
                  <th>PNL</th>
                  <th style={{ textAlign: 'right' }}>OUTCOME</th>
                </tr>
              </thead>
              <tbody>
                {[...(stats.recent_trades ?? [])].reverse().slice(0, 100).map((t, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #060a07', color: t.pnl >= 0 ? '#00ff88' : '#ff4444' }}>
                    <td style={{ padding: '4px 0', opacity: 0.5, fontFamily: 'monospace' }}>{t.time ?? 'N/A'}</td>
                    <td style={{ opacity: 0.5 }}>{t.step}</td>
                    <td style={{ fontWeight: 900 }}>{t.side}</td>
                    <td>{t.entry.toFixed(5)}</td>
                    <td>{t.exit.toFixed(5)}</td>
                    <td style={{ fontWeight: 900 }}>
                      {t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)}
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 900 }}>
                      <span style={{
                        padding: '1px 4px',
                        background: t.outcome === 'TP' ? 'rgba(0,255,136,0.1)' : t.outcome === 'SL' ? 'rgba(255,68,68,0.1)' : t.outcome === 'BE' ? 'rgba(0,204,255,0.1)' : 'rgba(255,255,255,0.05)',
                        borderRadius: '2px',
                        marginRight: '4px'
                      }}>
                        {t.outcome}
                      </span>
                      <span style={{ opacity: 0.35, fontSize: '7px' }}>
                        ({t.duration_min !== undefined ? `${t.duration_min.toFixed(1)}m` : `${t.duration}t`})
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.15, fontSize: '10px' }}>
              NO_TRADE_HISTORY
            </div>
          )}
        </div>
      </main>
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
