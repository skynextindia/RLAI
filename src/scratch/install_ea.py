import MetaTrader5 as mt5
import os
import shutil

def find_mt5_data_folder():
    print(">>> Locating MT5 Data Folder...")
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        return

    # This doesn't directly give the data path, but we can get it via terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        # The data path is usually in the AppData folder for the terminal instance
        data_path = terminal_info.data_path
        print(f"\n[FOUND] MT5 Data Path: {data_path}")
        
        experts_path = os.path.join(data_path, "MQL5", "Experts")
        print(f"[FOUND] MT5 Experts Path: {experts_path}")
        
        source_ea = r"D:\work\axon\RLBOT\src\mt5_bridge\ZeroMQ_Bridge.mq5"
        target_ea = os.path.join(experts_path, "ZeroMQ_Bridge.mq5")
        
        print(f"\n[ACTION] Copying EA to MT5...")
        try:
            shutil.copy2(source_ea, target_ea)
            print(f"SUCCESS: EA copied to {target_ea}")
            print("\n>>> NEXT STEPS IN MT5:")
            print("1. Right-click 'Expert Advisors' in the Navigator and select 'Refresh'.")
            print("2. Find 'ZeroMQ_Bridge' and drag it onto a BTCUSDm chart.")
            print("3. Ensure 'Allow Algo Trading' is checked in the EA common tab.")
        except Exception as e:
            print(f"ERROR: Could not copy EA: {e}")
            print(f"Please manually copy {source_ea} to {experts_path}")
    else:
        print("Could not retrieve terminal info.")

    mt5.shutdown()

if __name__ == "__main__":
    find_mt5_data_folder()
