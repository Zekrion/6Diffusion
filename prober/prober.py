import subprocess
import platform

class IPv6Prober:
    def __init__(self, timeout=1, count=1):
        """
        Initialize the IPv6 Prober.
        :param timeout: Timeout for the ping response in seconds.
        :param count: Number of ping attempts.
        """
        self.timeout = timeout
        self.count = count

    def is_reachable(self, ipv6_address):
        """
        Check if an IPv6 address is reachable.
        :param ipv6_address: The IPv6 address to probe.
        :return: True if reachable, False otherwise.
        """
        if platform.system().lower() != "windows":
            raise EnvironmentError("This script is designed for Windows.")

        try:
            result = subprocess.run(
                ["ping", "-6", "-n", str(self.count), "-w", str(self.timeout * 1000), ipv6_address],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            return "Reply from" in result.stdout
        except Exception as e:
            print(f"Error probing {ipv6_address}: {e}")
            return False

    def probe_list(self, ipv6_addresses):
        """
        Check the reachability of a list of IPv6 addresses.
        :param ipv6_addresses: List of IPv6 addresses to probe.
        :return: Dictionary with addresses as keys and reachability (True/False) as values.
        """
        results = {}
        for address in ipv6_addresses:
            results[address] = self.is_reachable(address)
        return results

# Example usage
if __name__ == "__main__":
    prober = IPv6Prober(timeout=1, count=2)
    ipv6_test_list = [
        "2001:4860:4860::8888",  # Google Public DNS IPv6
        "2606:4700:4700::1111"  # Cloudflare Public DNS IPv6
    ]
    results = prober.probe_list(ipv6_test_list)
    for addr, status in results.items():
        print(f"{addr} is {'reachable' if status else 'not reachable'}.")
