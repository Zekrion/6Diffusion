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

# Example usage
if __name__ == "__main__":
    prober = IPv6Prober(timeout=1, count=2)
    ipv6_test = "2001:4860:4860::8888"  # Example: Google Public DNS IPv6
    if prober.is_reachable(ipv6_test):
        print(f"{ipv6_test} is reachable.")
    else:
        print(f"{ipv6_test} is not reachable.")
