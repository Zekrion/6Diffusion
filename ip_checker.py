import sys
from prober import check_reachability

def main():
    if len(sys.argv) < 2:
        print("Usage: python ip_checker.py <IPv6_address> [-f <file>] [-r] [-w]")
        return
        
    target_ips = []
    input_file = None
    random_probes = False
    wait_time = 1
    
    # Parse command line arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '-f':
            if i+1 < len(args):
                input_file = args[i+1]
                i +=2
            else:
                print("Error: -f requires a file argument")
                return
        elif args[i] == '-r':
            random_probes = True
            i +=1
        elif args[i] == '-w':
            if i+1 < len(args):
                wait_time = int(args[i+1])
                i +=2
            else:
                print("Error: -w requires a time argument")
                return
        else:
            target_ips.append(args[i])
            i +=1
            
    # Read IPs from file if specified
    if input_file:
        try:
            with open(input_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                target_ips.extend([line for line in lines if line])
                
            if not target_ips:
                print("No valid IPv6 addresses found in file")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    # Check reachability for each IP
    results = []
    for ip in target_ips:
        try:
            result = check_reachability(ip, random_probes, wait_time)
            results.append(result)
        except ValueError as ve:
            print(f"Invalid IPv6 address: {ip}")
        except Exception as e:
            print(f"Error checking reachability for {ip}: {e}")

    # Display results
    if not results:
        return
    
    total_reachable = sum(1 for r in results if r['successful_probes'] > 0)
    total_ips = len(results)
        
    print("\nReachability Results:")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['ip_address']):
        status = "REACHABLE" if result['reachability'] > 0 else "UNREACHABLE"
        print(f"""
IP Address: {result['ip_address']}
Status: {status}
Probes Sent: {result['probes_sent']}
Successful Probes: {result['successful_probes']}
Reachability: {result['reachability']}%
Timestamp: {result['timestamp']:.2f}
""")
    
if __name__ == "__main__":
    main()
