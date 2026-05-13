//+------------------------------------------------------------------+
//|                                                ZeroMQ_Bridge.mq5 |
//+------------------------------------------------------------------+
#property copyright "AI Architect"
#property link      ""
#property version   "1.00"

#include <Zmq/Zmq.mqh> // Assumes https://github.com/dingmaotu/mql-zmq

Context context("mt5_ai");
Socket push_socket(context, ZMQ_PUSH);
Socket pull_socket(context, ZMQ_PULL);

void OnInit() {
   push_socket.connect("tcp://127.0.0.1:5555");
   pull_socket.connect("tcp://127.0.0.1:5556");
   EventSetMillisecondTimer(100);
   Print("ZMQ Bridge Initialized");
}

void OnTick() {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   long vol = iVolume(_Symbol, PERIOD_CURRENT, 0);
   
   string payload = StringFormat("{\"symbol\":\"%s\",\"ask\":%f,\"bid\":%f,\"volume\":%d}", _Symbol, ask, bid, vol);
   
   ZmqMsg msg(payload);
   push_socket.send(msg);
}

void OnTimer() {
   ZmqMsg msg;
   if(pull_socket.recv(msg, ZMQ_DONTWAIT)) {
      string data = msg.getData();
      Print("Received AI Signal: ", data);
      // Execution logic (OrderSend) to be implemented here
   }
}

void OnDeinit(const int reason) {
   EventKillTimer();
   push_socket.close();
   pull_socket.close();
   context.destroy();
   Print("ZMQ Bridge Deinitialized");
}
