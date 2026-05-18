import urllib.request
import json

try:
    res = urllib.request.urlopen('http://127.0.0.1:8080/telemetry')
    data = json.loads(res.read().decode())
    trades = data.get('recent_trades', [])
    print(f"Recent Trades: {len(trades)}")
    print(f"Current Step: {data.get('step')}")
    print("Last 5 Trades:")
    for t in trades[-5:]:
        pips = abs(t['entry'] - t['exit']) * 10000
        print(f"  Step {t['step']} {t['side']}: Entry {t['entry']:.5f} -> Exit {t['exit']:.5f} | PNL: {t['pnl']:.2f} ({t['outcome']}) | Pips Diff: {pips:.1f}")
except Exception as e:
    print(f"Error: {e}")
