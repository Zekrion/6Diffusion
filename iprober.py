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
        
        try:
            sock = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self.timeout)
            
            start_time = time.time()
            sent = sock.sendto(packet, (address, 1))  # Port doesn't matter for ICMP
            
            while True:
                ready = select.select([sock], [], [], self.timeout - (time.time() - start_time))
                if ready[0]:
                    data, addr = sock.recvfrom(1024)
                    end_time = time.time()
                    rtt = end_time - start_time
                    return {'address': address, 'rtt': rtt, 'reachable': True}
                
        except socket.timeout:
            # If timeout occurs without receiving a response
            return {'address': address, 'rtt': None, 'reachable': False}
            
        except socket.error as e:
            if e.errno == 10060:  # Connection timed out (specific to Windows)
                return {'address': address, 'rtt': None, 'reachable': False}
            print(f"Probe error for {address}: {e}")
            return {'address': address, 'rtt': None, 'reachable': False}
            
        finally:
            try:
                sock.close()
            except Exception as e:
                pass  # Ensure we don't let exceptions from close() bubble up
        
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
                if self.completed % 10 == 0:  # Print progress every 10 addresses to avoid spamming
                    print(f"\rScanned: {self.completed}/{num_addresses}", end='')
            except Exception as e:
                self._handle_error(addr, e)
                
        threads = []
        for addr in addresses:
            if not self.running:
                break
            # Limit number of active threads
            while len(threads) >= self.max_threads:
                threads.pop()  # Remove the oldest thread to make way for new ones
            thread = Thread(target=thread_function, args=(addr,))
            thread.start()
            threads.append(thread)
        
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

def generate_test_file():
    """Generate test_ipv6_addresses.txt with sample addresses"""
    import os
    
    test_addresses = [
        '::1/128',  # Loopback address (localhost)
        'fe80::a00:e000:1',  # link-local IPv6 address
        '2001:db8::8529b:af1e:8d58:f349/64',  # IPv6 global unicast
        'ff00::1',  # Multicast address (IPv6 link-local)
        '2001:0db8:85a3:0000:0000:8a2e:0370:7334',  # Another global unicast
        '::/128',   # All zeros (unreachable)
        'fe80::dead:beef:cafe:baad',  # Link-local address with random host suffix
        '2001:db8:1234:5678:9abc:def0:1234:5678/64'  # Another global unicast
    ]
    
    filename = 'test_ipv6_addresses.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(test_addresses))
        
if __name__ == '__main__':
    import sys, os
    
    # Generate test file if it doesn't exist
    if not os.path.exists('test_ipv6_addresses.txt'):
        generate_test_file()
    
    # Example usage:
    prober = IPv6Scanner(max_threads=4)
    print("\nRunning IPv6 probe tests...")
    prober.start_scan('test_ipv6_addresses.txt')
    time.sleep(2)  # Give it enough time to complete
    results = prober.get_results()
    print("\nTest Results:")
    for address in results['addresses']:
        reachable = '✓' if address['reachable'] else '✗'
        print(f"{address['address']} {reachable} (RTT: {address['rtt']:0.4f}s)")
    
    stats = results['stats']
    print("\nOverall Statistics:")
    print(f"Total probes: {stats['total_probes']}")
    print(f"Successful probes: {stats['successful_probes']}")
    print(f"Unresponsive count: {stats['unresponsive_count']}")
    print(f"Packet loss: {stats['packet_loss']}%")
    print(f"Scan duration: {stats['scan_duration']:0.2f}s")
