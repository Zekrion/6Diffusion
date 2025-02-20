import socket
import select
import random
import time
from threading import Thread

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

class IPv6Scanner(IPv6Prober):
    """
    Enhanced IPv6 scanning class that provides parallel probing and improved 
    statistics.
    
    Inherits from IPv6Prober and adds:
    - Parallel scanning using threads
    - Better error handling
    - Statistics collection
    - More detailed reporting
    
    Args:
        send_rate (int): Packets per second (default: 1)
        retries (int): Number of retry attempts per address (default: 3)
        timeout (float): Timeout in seconds for each probe (default: 2.0)
        max_threads (int): Maximum number of parallel probes (default: 8)
    """
    def __init__(self, send_rate=1, retries=3, timeout=2.0, max_threads=8):
        super().__init__(send_rate, retries, timeout)
        self.max_threads = max_threads
        self.results = []
        self.completed = 0
        self.start_time = None
        self.running = False
        
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
    
    def start_scan(self, filename):
        """
        Start scanning all addresses in the given file with parallel threads.
        
        Returns:
            None (use get_results() to retrieve final results)
        """
        self.results = []
        self.completed = 0
        self.start_time = time.time()
        self.running = True
        
        # Create and start threads for each address
        addresses = list(self.read_ipv6_list(filename))
        num_addresses = len(addresses)
        
        def thread_function(addr):
            try:
                result = self.probe_address(addr)
                self.results.append(result)
                self.completed += 1
                print(f"\rScanned: {self.completed}/{num_addresses}", end='')
            except Exception as e:
                self._handle_error(addr, e)
                
        for addr in addresses:
            if self.running and len(Thread.active()) < self.max_threads:
                Thread(target=thread_function, args=(addr,)).start()
        
    def stop_scan(self):
        """
        Stop the scanning process gracefully.
        
        Returns:
            None
        """
        self.running = False
        
    def get_results(self):
        """
        Get the final results of the scan.
        
        Returns:
            dict: Contains overall statistics and per-address results
        """
        if not self.results:
            return {'status': 'error', 'message': 'No scan completed'}
            
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Calculate statistics
        reachable = [r for r in self.results if r['reachable']]
        unresponsive = [r for r in self.results if not r['reachable']]
        
        stats = {
            'total_probes': len(self.results),
            'successful_probes': len(reachable),
            'unresponsive_count': len(unresponsive),
            'min_rtt': min(r['rtt'] for r in reachable) if reachable else None,
            'avg_rtt': sum(r['rtt'] for r in reachable)/len(reachable) if reachable else None,
            'max_rtt': max(r['rtt'] for r in reachable) if reachable else None,
            'packet_loss': (len(unresponsive) / len(self.results)) * 100 if self.results else 0.0,
            'scan_duration': total_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
        
        return {
            'stats': stats,
            'addresses': self.results
        }

# Example usage:
# prober = IPv6Prober(send_rate=2, retries=3, timeout=2)
# results = prober.probe_all('ipv6-addresses.txt')
