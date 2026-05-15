import MetaTrader5 as mt5

def force_sl_patch():
    print(">>> Initializing Emergency SL Patch...")
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        return

    symbol = "BTCUSDm"
    positions = mt5.positions_get(symbol=symbol)
    
    if not positions or len(positions) == 0:
        print(f"No active positions found for {symbol} to patch.")
        return

    for pos in positions:
        if pos.sl == 0.0:
            print(f"Found unprotected position: Ticket {pos.ticket} @ {pos.price_open}")
            
            # Calculate a safe 0.5% SL
            digits = mt5.symbol_info(symbol).digits
            sl_dist = pos.price_open * 0.005
            new_sl = round(pos.price_open - sl_dist if pos.type == mt5.ORDER_TYPE_BUY else pos.price_open + sl_dist, digits)
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp # Keep existing TP if any
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"SUCCESS: Injected SL {new_sl} into Ticket {pos.ticket}")
            else:
                print(f"FAILURE: Could not inject SL. Error: {result.retcode} - {result.comment}")
        else:
            print(f"Position {pos.ticket} already has SL: {pos.sl}")

    mt5.shutdown()

if __name__ == "__main__":
    force_sl_patch()
