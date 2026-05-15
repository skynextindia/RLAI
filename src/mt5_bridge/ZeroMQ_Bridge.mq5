#property copyright "Axon AI"
#property version   "5.30"
#property strict

input string ServerURL = "http://127.0.0.1:8000/api/telemetry";

void OnTick() {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   string payload = StringFormat("{\"symbol\":\"%s\",\"ask\":%f,\"bid\":%f}", _Symbol, ask, bid);
   char data[], result[];
   string headers;
   StringToCharArray(payload, data);
   int res = WebRequest("POST", ServerURL, "Content-Type: application/json", 10, data, result, headers);
   if(res == -1) Print("Error in WebRequest: ", GetLastError());
}
