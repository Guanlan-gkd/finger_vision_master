#! /bin/sh
echo "-> Setting up environment required to connect to raspberry pi"
ifconfig enx00e04c685113 10.42.0.1
iptables -F
iptables -t nat -A POSTROUTING -o wlp0s20f3 -j MASQUERADE
iptables -A FORWARD -i wlp0s20f3 -o enx00e04c685113 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i enx00e04c685113 -o wlp0s20f3 -j ACCEPT