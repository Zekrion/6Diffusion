import socket
from scapy.layers.l2tp import icmp6
from scapy.packet import Packet

def icmp_probe_function(target_address):
    """
    Simple IPv6 ICMP probe function that sends Echo Request and checks response.
    Returns True if reachable, False otherwise.
    """
    try:
        # Create a raw socket for IPv6
        s = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
        
        # Craft an ICMPv6 Echo Request packet
        icmp_pkt = icmp6.ICMPv6EchoRequest(
            dst=target_address,
            payload=b'ProberTest'
        )
        
        # Send the packet and get response
        reply = icmp_pkt.send()
        
        if not isinstance(reply, Packet):
            return False
            
        if reply.icmp6_type == icmp6.ICMPv6_EchoReply:
            print(f"Successfully probed: {target_address}")
            s.close()
            return True
        
    except Exception as e:
        print(f"Probe failed to {target_address}: {str(e)}")
        return False
    
    finally:
        if 's' in locals():
            s.close()
    
    return False

def multiple_address_probe(addresses):
    """
    Probes a list of IPv6 addresses and returns list of reachable ones.
    """
    reachable = []
    for addr in addresses:
        if icmp_probe_function(addr):
            reachable.append(addr)
    return reachable

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prober.py <address1> [address2] ...")
        sys.exit(1)
    
    addresses = sys.argv[1:]
    print(f"Probing addresses: {', '.join(addresses)}")
    reachable = multiple_address_probe(addresses)
    if reachable:
        print("\nReachable hosts:")
        for addr in reachable:
            print(addr)
    else:
        print("No reachable hosts found.")
