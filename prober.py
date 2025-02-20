from scapy.all import IPv6, ICMPv6EchoRequest
import time
import random

def send_icmp6_probe(target_ip):
    """Send a single ICMPv6 echo request probe and return the response"""
    # Create the ICMPv6 echo request packet
    icmp_pkt = ICMPv6EchoRequest()
    ip_pkt = IPv6(dst=target_ip) / icmp_pkt
    
    # Send the packet with timeout 1 second
    try:
        response = ip_pkt.send()
        if response:
            return True
        return False
    except Exception as e:
        print(f"Error probing {target_ip}: {e}")
        return None

def check_reachability(target_ip, probes=5, interval=1):
    """Check reachability by sending multiple probes with a given interval"""
    success = 0
    for _ in range(probes):
        try:
            if send_icmp6_probe(target_ip):
                success += 1
        except Exception as e:
            print(f"Probe failed: {e}")
        time.sleep(interval)
    
    reachability = (success / probes) * 100
    return {
        'ip_address': target_ip,
        'probes_sent': probes,
        'successful_probes': success,
        'reachability': round(reachability, 2),
        'timestamp': time.time()
    }

if __name__ == "__main__":
    # Example usage:
    target = "2001:db8::1"
    result = check_reachability(target)
    print(f"Reachability for {result['ip_address']}: {result['reachability']}%")
