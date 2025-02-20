import socket
import select
import random
import time

class IPv6Prober:
    """
    A class to probe IPv6 addresses with configurable parameters.
    
    Args:
        send_rate (int): Packets per second (default: 1)
        retries (int): Number of retry attempts per address (default: 3)
        timeout (float): Timeout in seconds for each probe (default: 2.0)
    """
    def __init__(self, send_rate=1, retries=3, timeout=2.0):
        self.send_rate = send_rate
        self.retries = retries
        self.timeout = timeout
        
    def read_ipv6_list(self, filename):
        """Read IPv6 addresses from file"""
        with open(filename, 'r') as f:
            for line in f:
                addr = line.strip()
                if addr:
                    yield addr
                    
    def probe_address(self, address):
        """
        Probe a single IPv6 address.
        
        Returns:
            dict: Contains 'address', 'rtt' (round trip time), and 'reachable'
        """
        packet = bytes.fromhex('0800')  # ICMPv6 Echo Request
        for attempt in range(self.retries):
            try:
                sock = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
                sock.setblocking(False)
                
                start_time = time.time()
                sent = sock.sendto(packet, (address, 1))  # Port doesn't matter for ICMP
                
                ready = select.select([sock], [], [], self.timeout)
                if ready[0]:
                    data, addr = sock.recvfrom(1024)
                    end_time = time.time()
                    rtt = end_time - start_time
                    return {'address': address, 'rtt': rtt, 'reachable': True}
                
            except socket.error as e:
                print(f"Probe error for {address}: {e}")
            
            finally:
                sock.close()
        
        return {'address': address, 'rtt': None, 'reachable': False}
    
    def probe_all(self, filename):
        """
        Probe all addresses in the given file.
        
        Returns:
            list: List of probe results
        """
        results = []
        for addr in self.read_ipv6_list(filename):
            result = self.probe_address(addr)
            results.append(result)
            
            # Rate limiting
            time.sleep(max(0, 1.0 / self.send_rate - (time.time() - result['rtt'])))
            if result['reachable']:
                time.sleep(random.uniform(0.5, 1.5))  # Randomize timing between probes
            
        return results

# Example usage:
# prober = IPv6Prober(send_rate=2, retries=3, timeout=2)
# results = prober.probe_all('ipv6-addresses.txt')
