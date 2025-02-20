import socket
import time
from ipaddress import IPv6Address

def send_icmp6_probe(target_ip):
    """Send a single ICMPv6 echo request probe and return the response"""
    try:
        # Create IPv6 socket
        sock = socket.socket(socket.AF_INET6, socket.SOCK_ICMPV6)
        
        # Set timeout for the socket
        sock.settimeout(1.0)
        
        # Send empty echo request (ICMPv6 type 1, code 0)
        sock.sendto(b'', (target_ip, 0))
        
        # Close the socket
        sock.close()
        
        return True
        
    except Exception as e:
        print(f"Error probing {target_ip}: {e}")
        return False

def check_reachability(target_ip, probes=5, interval=1):
    """Check reachability by sending multiple probes with a given interval"""
    success = 0
    
    try:
        # Validate IPv6 address
        IPv6Address(target_ip)
        
        for _ in range(probes):
            try:
                if send_icmp6_probe(target_ip):
                    success += 1
                time.sleep(interval)
                
            except socket.timeout:
                print(f"Timeout probing {target_ip}")
                continue
                
            except Exception as e:
                print(f"Probe failed: {e}")
                continue
                
    except ValueError:
        raise ValueError(f"Invalid IPv6 address: {target_ip}")

    reachability = (success / probes) * 100 if probes > 0 else 0
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
