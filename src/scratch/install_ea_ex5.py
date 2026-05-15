import MetaTrader5 as mt5
import os
import shutil

def install_compiled_ea():
    print(">>> Locating MT5 Data Folder...")
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        return

    terminal_info = mt5.terminal_info()
    if terminal_info:
        data_path = terminal_info.data_path
        experts_path = os.path.join(data_path, "MQL5", "Experts")
        
        # Source files
        source_ex5 = r"D:\work\axon\RLBOT\src\mt5_bridge\ZeroMQ_Bridge.ex5"
        source_mq5 = r"D:\work\axon\RLBOT\src\mt5_bridge\ZeroMQ_Bridge.mq5"
        
        # Targets
        target_ex5 = os.path.join(experts_path, "ZeroMQ_Bridge.ex5")
        target_mq5 = os.path.join(experts_path, "ZeroMQ_Bridge.mq5")
        
        print(f"\n[ACTION] Deploying Compiled EA to MT5...")
        try:
            # Copy Compiled Binary (Bypasses dependency issues)
            if os.path.exists(source_ex5):
                shutil.copy2(source_ex5, target_ex5)
                print(f"SUCCESS: EX5 Binary deployed to {target_ex5}")
            
            # Copy Source Code
            if os.path.exists(source_mq5):
                shutil.copy2(source_mq5, target_mq5)
                print(f"SUCCESS: MQ5 Source deployed to {target_mq5}")
                
            print("\n>>> CRITICAL NEXT STEPS:")
            print("1. Right-click 'Expert Advisors' in MT5 Navigator -> REFRESH.")
            print("2. You should now see 'ZeroMQ_Bridge' (with a blue icon).")
            print("3. Drag it onto a BTCUSDm chart.")
            print("4. Ensure 'Allow Algo Trading' is checked in Common Tab.")
        except Exception as e:
            print(f"ERROR: Deployment failed: {e}")
    else:
        print("Could not retrieve terminal info.")

    mt5.shutdown()

if __name__ == "__main__":
    install_compiled_ea()
