import zmq
import json
import threading
from datetime import datetime

class MT5ZMQClient:
    def __init__(self, pull_port=5555, push_port=5556):
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.LINGER, 0)
        self.pull_socket.bind(f"tcp://127.0.0.1:{pull_port}")
        
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.LINGER, 0)
        self.push_socket.bind(f"tcp://127.0.0.1:{push_port}")
        print(f">>> [ZMQ] Neural Port {pull_port} is now OPEN and BINDED.")
        
    def stream_market_data(self, callback):
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)
        
        while True:
            try:
                # Poll with 1s timeout to allow thread to check for signals/exit
                socks = dict(poller.poll(1000))
                if self.pull_socket in socks:
                    # Drain the socket queue completely to prevent lag
                    while True:
                        try:
                            message = self.pull_socket.recv_string(zmq.DONTWAIT)
                            data = json.loads(message)
                            data['timestamp'] = datetime.now().isoformat()
                            callback(data)
                        except zmq.Again:
                            break # Queue empty
            except Exception as e:
                if not isinstance(e, zmq.Again):
                    print(f"ZMQ Error: {e}")
            
    def send_order(self, action, symbol, volume, sl=0.0, tp=0.0):
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("MT5 Init Failed for Execution")
            return
            
        if action == "CLOSE_ALL":
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    tick = mt5.symbol_info_tick(symbol)
                    deal_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    price = tick.bid if deal_type == mt5.ORDER_TYPE_SELL else tick.ask
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": deal_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": 999999,
                        "comment": "AI Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    print(f"[MT5 VERIFY] CLOSE retcode={result.retcode} order={result.order} sl_sent={request.get('sl',0)} tp_sent={request.get('tp',0)}")
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Close Failed: {result.comment}")
                    else:
                        print(f">>> [MT5 TERMINAL] Successfully closed {symbol} position")
            return

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if tick is None or symbol_info is None:
            print(f"Failed to get info for {symbol}")
            return
            
        digits = symbol_info.digits
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": round(price, digits),
            "sl": round(float(sl), digits) if sl > 0 else 0.0,
            "tp": round(float(tp), digits) if tp > 0 else 0.0,
            "deviation": 20,
            "magic": 999999,
            "comment": "Axon RL Entry",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        print(f"[MT5 VERIFY] ENTRY retcode={result.retcode} order={result.order} sl_sent={request['sl']} tp_sent={request['tp']}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order Failed: {result.retcode} - {result.comment}")
            return None
        else:
            print(f">>> [MT5 TERMINAL] Successfully punched {action} {volume} lots on {symbol}")
            return result.order
